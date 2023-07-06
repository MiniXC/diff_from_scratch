import argparse
import inspect
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Optional
import json

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, whoami
from tqdm.auto import tqdm
import numpy as np
from vocex import Vocex

# from augmentations import wave_augmentation_func

import diffusers
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel # DDPMPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from pipeline import DDPMPipeline
from pipeline import DDPMPipeline

from speech_collator import SpeechCollator


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

class PhoneEmbedding(torch.nn.Module):
    def __init__(self, phone2idx, embedding_dim=80):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(phone2idx), embedding_dim)
        self.phone2idx = phone2idx

    def forward(self, phone):
        return self.embedding(phone)
    
class Projection(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, condition):
        return self.linear(condition)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_type",
        type=str,
        default="vocex",
        help="Can be 'mel' or 'vocex'."
    )
    parser.add_argument(
        "--cut_epochs_short",
        type=int,
        default=-1,
        help="Cut epochs short for debugging."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="vocex-baseline",
        help="The name of the model to train.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--is_conditional",
        default=True,
        action="store_true",
        help="Whether the model should be trained in conditional mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="conditional_ddpm",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=1, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=2, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mel_mean_norm",
        type=float,
        default=15.0,
        help="The mean value to normalize the mel spectrograms with.",
    )
    parser.add_argument(
        "--mel_std_norm",
        type=float,
        default=29.0,
        help="The standard deviation value to normalize the mel spectrograms with.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=50)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="sigmoid")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--log_loss_every",
        type=int,
        default=100,
        help="Log the average training loss every X updates. Set to 0 to disable.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    if args.train_type == "mel":
        resolution_x = 512
        resolution_y = 80
    elif args.train_type == "vocex":
        resolution_x = 384
        resolution_y = 16

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        wandb.init(project="diffusers", config=vars(args), name=args.model_id)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    if args.model_config_name_or_path is None:
        # 128, 
        # 128,
        # 256,
        # 256,
        # 512,
        # 512
        if not args.is_conditional:
            model = UNet2DModel(
                sample_size=(resolution_x, resolution_y),
                in_channels=1,
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(
                    64,
                    64,
                    128,
                    256,
                    512,
                ),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    # "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    # "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                # encoder_hid_dim=80,
            )
        else:
            if args.train_type == "mel":
                in_channels = 1 + 4 + 4 # noise, phone, vocex
            elif args.train_type == "vocex":
                in_channels = 1 + 4 # noise, phone
            model = UNet2DConditionModel(
                sample_size=(resolution_x, resolution_y),
                in_channels=in_channels,
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(
                    64,
                    64,
                    128,
                    256,
                    512,
                ),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    # "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    # "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                cross_attention_dim=128,
            )
    else:
        if not args.is_conditional:
            config = UNet2DModel.load_config(args.model_config_name_or_path)
            model = UNet2DModel.from_config(config)
        else:
            config = UNet2DConditionModel.load_config(args.model_config_name_or_path)
            model = UNet2DConditionModel.from_config(config)

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = load_dataset("cdminix/libritts-r-aligned")
    ds_train = dataset["train"]
    ds_val = dataset["dev"]

    with open("data/speaker2idx.json", "r") as f:
        speaker2idx = json.load(f)
    with open("data/phone2idx.json", "r") as f:
        phone2idx = json.load(f)

    if args.is_conditional:
        if args.train_type == "mel":
            phone_embedding = PhoneEmbedding(phone2idx, embedding_dim=80*4)
            vocex_projection = Projection(16, 80*4)
        elif args.train_type == "vocex":
            phone_embedding = PhoneEmbedding(phone2idx, embedding_dim=16*4)
        condition_projection = Projection(80, 128)

    vocex_model = Vocex.from_pretrained("cdminix/vocex").model.to("cpu")

    collator = SpeechCollator(
        speaker2idx=speaker2idx,
        phone2idx=phone2idx,
        use_speaker_prompt=True,
        overwrite_max_length=True,
        vocex_model=vocex_model,
        expand_seq=args.train_type == "mel",
    )

    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.train_batch_size,
        collate_fn=collator.collate_fn,
        num_workers=0,#args.dataloader_num_workers,
        shuffle=True,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        ds_val,
        batch_size=args.eval_batch_size,
        collate_fn=collator.collate_fn,
        num_workers=0,
        shuffle=False,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    if args.is_conditional:
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, phone_embedding, condition_projection = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, phone_embedding, condition_projection
        )
        if args.train_type == "mel":
            vocex_projection = accelerator.prepare(
                vocex_projection
            )
    else:
        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    losses = []

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        step = -1
        for batch in train_dataloader:
            step += 1

            if args.cut_epochs_short != -1 and step >= args.cut_epochs_short:
                break

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            
            # clean_images, _ = ptwt.cwt(batch["phone_durations"].to(torch.int32), torch.arange(1, 81, 0.1), "mexh")
            # clean_images = clean_images.transpose(0, 1) # [bsz, 32, 512]
            if args.train_type == "mel":
                clean_images = batch["mel"].unsqueeze(1)
            elif args.train_type == "vocex":
                clean_images = batch["vocex"].unsqueeze(1)
            #batch["mel"].unsqueeze(1) # [bsz, 1, 512, 80]

            if args.train_type == "mel":
                attn_mask = batch["frame_mask"]
            elif args.train_type == "vocex":
                attn_mask = batch["phone_mask"]

            # pad to bsz, 1, 512, 128 from bsz, 1, 512, 80
            # clean_images = F.pad(clean_images, (0, 0, 0, 0, 0, 48), "constant", 0)

            # mean & std norm
            if args.train_type == "mel":
                clean_images = (clean_images - args.mel_mean_norm) / args.mel_std_norm
            # get to mean 0.5 and std 0.5
            clean_images = clean_images * 0.5 # * 0.5 + 0.5

            condition = batch["speaker_prompt_mel"] # [bsz, length, 80]

            condition = (condition - args.mel_mean_norm) / args.mel_std_norm
            # get to mean 0.5 and std 0.5
            condition = condition * 0.5 # * 0.5 + 0.5

            enc_attn_mask = condition.sum(dim=-1) != 0 # [bsz, length]

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)


            if args.is_conditional:
                if args.train_type == "mel":
                    phone = batch["phones"]
                    phone = phone_embedding(phone) # [bsz, length, 80*4]
                    phone = phone.reshape(bsz, 4, -1, 80)
                    vocex = batch["vocex"]
                    vocex = vocex_projection(vocex)
                    vocex = vocex.reshape(bsz, 4, -1, 80)
                    # combine noise and phone
                    noisy_images = torch.cat([noisy_images, phone, vocex], dim=1)
                elif args.train_type == "vocex":
                    phone = batch["phones"]
                    phone = phone_embedding(phone)
                    phone = phone.reshape(bsz, 4, -1, 16)
                    noisy_images = torch.cat([noisy_images, phone], dim=1)
                # project condition
                condition = condition_projection(condition)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if args.is_conditional:
                    model_output = model(
                        noisy_images,
                        timesteps,
                        condition,
                        #attention_mask=attn_mask,
                        encoder_attention_mask=enc_attn_mask
                    ).sample
                else:
                    model_output = model(
                        noisy_images,
                        timesteps,
                    ).sample

                if args.prediction_type == "epsilon":
                    loss = F.mse_loss(model_output, noise, reduction="none")
                    loss = loss * attn_mask.unsqueeze(-1)
                    loss = loss.mean()
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.log_loss_every != 0:
                losses.append(loss)
                if global_step % args.log_loss_every == 0:
                    logs["loss"] = torch.mean(torch.tensor(losses)).item()
                    losses = []
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
            unet = accelerator.unwrap_model(model)

            if args.is_conditional:
                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                    conditional=True,
                    is_mel=args.train_type == "mel",
                )
            else:
                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                    conditional=False,
                )

            # run pipeline in inference (sample random noise and denoise)
            if args.is_conditional:
                # get conditions
                for batch in eval_dataloader:
                    condition = batch["speaker_prompt_mel"]
                    condition = (condition - args.mel_mean_norm) / args.mel_std_norm
                    # get to mean 0.5 and std 0.5
                    condition = condition * 0.5 # * 0.5 + 0.5
                    with torch.no_grad():
                        condition = condition_projection(condition)
                    enc_attn_mask = condition.sum(dim=-1) != 0
                    val_phone = phone_embedding(batch["phones"])
                    if args.train_type == "vocex":
                        images = pipeline(
                            batch_size=args.eval_batch_size,
                            num_inference_steps=args.ddpm_num_inference_steps,
                            output_type="numpy",
                            cond=condition,
                            phones=val_phone,
                            encoder_attention_mask=enc_attn_mask,
                        ).images
                    elif args.train_type == "mel":
                        val_vocex = vocex_projection(batch["vocex"])
                        images = pipeline(
                            batch_size=args.eval_batch_size,
                            num_inference_steps=args.ddpm_num_inference_steps,
                            output_type="numpy",
                            cond=condition,
                            phones=val_phone,
                            vocex=val_vocex,
                            encoder_attention_mask=enc_attn_mask,
                        ).images
                    break
            else:
                images = pipeline(
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                ).images

            images = accelerator.gather_for_metrics(images).cpu().numpy()
            phones = accelerator.gather_for_metrics(batch["phones"]).cpu().numpy()
            phone_mask = accelerator.gather_for_metrics(batch["phone_mask"]).cpu().numpy()

            # denormalize the images and save to tensorboard
            # images_processed = (images * 255).round().astype("uint8")
            images_processed = images - images.min()
            images_processed = images_processed / images_processed.max()
            images_processed = (images_processed * 255).round().astype("uint8")

            accelerator.wait_for_everyone()

            if args.logger == "tensorboard":
                if is_accelerate_version(">=", "0.17.0.dev0"):
                    tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                else:
                    tracker = accelerator.get_tracker("tensorboard")
                tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
            elif args.logger == "wandb":
                # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                if not args.is_conditional:
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img.T) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )
                else:
                    images = []
                    for i in range(images_processed.shape[0]):
                        img_masked = images_processed[i][phone_mask[i]]
                        # normalize to 0, 1
                        img_masked = img_masked - img_masked.min()
                        img_masked = img_masked / img_masked.max()
                        images.append(wandb.Image(img_masked.T))
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": images, "epoch": epoch},
                        step=global_step,
                    )

            accelerator.wait_for_everyone()

            del unet, pipeline, images, images_processed

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                accelerator.wait_for_everyone()

                # save the model
                unet = accelerator.unwrap_model(model)

                pipeline = DDPMPipeline(
                    unet=unet,
                    scheduler=noise_scheduler,
                )

                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

                del unet

            accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    main()
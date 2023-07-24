from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from diffusers.utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DDPMPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler, conditional=False, is_mel=False, seed=None, is_conformer=False, device=None):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.conditional = conditional
        self.is_mel = is_mel
        self.seed = seed
        self.is_conformer = is_conformer
        self._device = device

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        cond: Optional[torch.Tensor] = None,
        phones: Optional[torch.Tensor] = None,
        vocex: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        if self.seed is not None:
            # set seed and make deterministic
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            if self.is_conformer:
                image_shape = (
                    batch_size,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
        else:
            if self.is_conformer:
                image_shape = (batch_size, *self.unet.config.sample_size)
            else:
                image_shape = (batch_size, 1, *self.unet.config.sample_size)

        if self._device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape)
            image = image.to(self._device)
        else:
            image = randn_tensor(image_shape, device=self._device)

        if self.conditional:
            if self.is_mel:
                height = 80
            else:
                height = 16
            width = self.unet.config.sample_size[0]
            if not self.is_conformer:
                phones = phones.reshape(image_shape[0], width, height)
                # combine noise and phone
                if vocex is None:
                    image = torch.cat([image, phones], dim=1)
                else:
                    vocex = vocex.reshape(image_shape[0], width, height)
                    image = torch.cat([image, phones, vocex], dim=1)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if self.conditional:
                if self.is_conformer:
                    bsz = image.shape[0]
                    t_condition = torch.cat([phones, vocex], dim=2)
                    timesteps = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    timesteps = timesteps.to(image.device)
                    noisy_images = image.reshape(bsz, -1, 80)
                    model_output = self.unet(
                        noisy_images,
                        timesteps,
                        t_condition,
                        cond,
                        cond.sum(dim=-1) != 0
                    )
                else:
                    model_output = self.unet(
                        image, 
                        t,
                        cond,
                        encoder_attention_mask=encoder_attention_mask,
                    ).sample
            else:
                model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image).prev_sample

        if not self.is_conformer:
            image = image[:, 0]

        return image
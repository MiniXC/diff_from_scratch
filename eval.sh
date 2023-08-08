accelerate launch training.py \
--wandb_mode=offline \
--model_type=conformer \
--model_id=conformer_postnet_16l_fix2 \
--eval_only \
--load_from_checkpoint=bff_tts/checkpoint-45936 \
--ddpm_num_inference_steps=10 \
--eval_batch_size=1
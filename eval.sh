accelerate launch training.py \
--wandb_mode=offline \
--model_type=conformer \
--model_id=conformer_postnet_16l_fix2 \
--eval_only \
--load_from_checkpoint=conformer_postnet_16l_fix3/checkpoint-48807
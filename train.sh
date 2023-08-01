accelerate launch training.py \
--wandb_mode=offline \
--model_type=conformer \
--model_id=conformer_postnet_16l_fix_eval \
--load_from_checkpoint=conformer_postnet_16l_fix/checkpoint-287100 \
--eval_only
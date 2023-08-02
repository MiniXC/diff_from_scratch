accelerate launch training.py \
--wandb_mode=offline \
--model_type=conformer \
--model_id=conformer_postnet_16l_fix2 \
--load_from_checkpoint=conformer_postnet_16l_fix2/checkpoint-20097 \
--no_eval

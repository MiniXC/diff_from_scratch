accelerate launch training.py \
--wandb_mode=offline \
--model_type=conformer \
--model_id=conformer_postnet_16l_fix3 \
--no_eval \
--loss_mode="mse"

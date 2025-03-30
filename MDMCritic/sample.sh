export CUDA_VISIBLE_DEVICES=0

exp_name="mdmft_physcritic_test1"
python -m train.tune_mdm \
    --dataset humanact12 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 \
    --resume_checkpoint ./save/finetuned/mdmft_physcritic_test1/model350100.pt \
    --critic_model_path ./pretrained/physcritic_lossplcc_perprompt_phys0.3.pth \
    --device 0 \
    --num_steps 1200 \
    --save_interval 100 \
    --critic_scale 1e-3 --kl_scale 1\
    --ddim_sampling \
    --eval_during_training \
    --sample_when_eval \
    --batch_size 64 --lr 1e-5 \
    --denoise_lower 700 --denoise_upper 900 \
    --use_kl_loss \
    --save_dir "save/finetuned/${exp_name}"  --overwrite\

# tmux 6
# python -m train.sample_and_save \
#     --dataset humanact12 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 \
#     --resume_checkpoint ./save/finetuned/mdmft_physcritic_test1/model350100.pt \
#     --critic_model_path ./pretrained/physcritic_lossplcc_perprompt_phys0.3.pth \
#     --device 0 \

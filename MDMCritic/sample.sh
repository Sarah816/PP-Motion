export CUDA_VISIBLE_DEVICES=2

exp_name="mdmft_physcritic_test1"

python -m train.sample_and_save \
    --dataset humanact12 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 \
    --resume_checkpoint ./save/finetuned/${exp_name}/model350100.pt \
    --critic_model_path ./pretrained/physcritic_lossplcc_perprompt_phys0.3.pth \
    --device 0 --save_dir "save/finetuned/${exp_name}" \
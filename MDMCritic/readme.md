# Using PP-Motion Model to Finetune MDM

## Preparation

1. Download MDM model (pretrained on HumanAct12) and place it in `./save/humanact12/model000350000.pt`
2. Prepare PP-Motion model (or your critic model) and place it in `./pretrained/critic.pth`

## Training

```bash
bash finetune.sh
```

Or run manually:

```bash
exp_name="mdmft_physcritic_test1"
python -m train.tune_mdm \
    --dataset humanact12 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 \
    --resume_checkpoint ./save/humanact12/model000350000.pt \
    --critic_model_path ./pretrained/critic.pth \
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
```

## Sample Motion

After finetuning, you can sample motions using your finetuned model. The following script gives an example of using the finetuned model at `./save/finetuned/mdmft_physcritic/model350100.pt` (finetuned 100 steps) to generate motions for all HumanAct12 evaluation dataset (1190 motions).

```bash
bash sample.sh
```

Or run manually:

```bash
exp_name="mdmft_physcritic_test1"

python -m train.sample_and_save \
    --dataset humanact12 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 \
    --resume_checkpoint ./save/finetuned/${exp_name}/model350100.pt \
    --critic_model_path ./pretrained/physcritic_lossplcc_perprompt_phys0.3.pth \
    --device 0 --save_dir "save/finetuned/${exp_name}" \
```

Please change `exp_name` to your finetuning directory.

This will automatically save generated motion data to `save/gen/uncriticed-motions-step{step}.pth`, and critic scores to `save/gen/uncriticed-critics-step{step}.pth`.
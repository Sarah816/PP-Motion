# MotionCritic: critic_correct_org_lr3.5e-5_singlegpu
# Ours: norm_lossplcc_perprompt_phys0.3
# MSE loss: norm_lossmse_exp2_phys0.3

# NOTE: Put model {checkpoint}.pth in directory output/{exp_name}/
export CUDA_VISIBLE_DEVICES=5
python metric/metrics.py \
    --mode mdmval \
    --exp_name norm_lossplcc_perprompt_phys0.3 \
    --checkpoint checkpoint_latest\
    --calc_baseline_metric \
    # --calc_phys_metric \
    # --calc_gt_metric \
    # --calc_perprompt
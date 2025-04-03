# "critic_correct_org_lr3.5e-5_singlegpu"
# "norm_lossplcc_perprompt_phys0.3"
export CUDA_VISIBLE_DEVICES=0
python metric/metrics.py \
    --mode mdm \
    --exp_name "norm_lossplcc_perprompt_phys0.3" \
    # --device_id "1"
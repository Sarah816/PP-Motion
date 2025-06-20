# --dataset "corrected"  --big_model 这个设置不能变

python train.py --exp_name norm_lossplcc_perprompt_phys0.2 \
    --dataset "corrected" --big_model \
    --learning_rate 2e-5 --batch_size 64  --lr_decay \
    --save_latest --save_checkpoint --gpu_indices "0,1" \
    --loss_type "plcc" --enable_phys --phys_coef 0.2

python train.py --exp_name critic_correct_org_lr3.5e-5_singlegpu \
    --dataset "corrected" --big_model \
    --learning_rate 2e-5 --batch_size 64  --lr_decay \
    --save_latest --save_checkpoint --gpu_indices "1,2" \
    --loss_type "plcc" --enable_phys --phys_coef 0.3
    
# --dataset "corrected"  --big_model 这个设置不能变

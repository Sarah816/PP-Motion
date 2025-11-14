python train.py --exp_name norm_lossplcc_perprompt_phys0.2_reproduce \
    --big_model --learning_rate 2e-5 --batch_size 64  --lr_decay \
    --save_latest --save_checkpoint --gpu_indices "6" \
    --loss_type "plcc" --enable_phys --phys_coef 0.2 \
    

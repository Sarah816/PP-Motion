python train.py --exp_name pp-motion \
    --big_model --learning_rate 2e-5 --batch_size 64  --lr_decay \
    --save_latest --save_checkpoint \
    --loss_type "plcc" --enable_phys --phys_coef 0.2 \
    

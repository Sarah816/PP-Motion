python train.py --exp_name criticphys_mdm_lr5e-6 \
    --dataset mdm \
    --learning_rate 5e-6 --batch_size 64\
    --save_latest --lr_decay --big_model --gpu_indices "4,7" 

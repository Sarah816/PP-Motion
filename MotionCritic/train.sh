python train.py --exp_name criticphys_2e-3 \
    --dataset mdm \
    --learning_rate 2e-3 --batch_size 64\
    --save_latest --lr_decay --big_model --gpu_indices "4,3"

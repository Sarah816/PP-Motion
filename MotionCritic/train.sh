python train.py --gpu_indices 5 --exp_name critic_mdmfull_lr2e-5 \
    --dataset mdmfull_shuffle \
    --learning_rate 2e-5 --batch_size 64\
    --save_latest --lr_decay --big_model --debug

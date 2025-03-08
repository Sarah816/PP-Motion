python train.py --gpu_indices 3,4 --exp_name critic_mdmfull_seed3407 \
    --dataset mdm \
    --learning_rate 2e-3 --batch_size 64 --seed 3407\
    --save_latest --lr_decay --big_model

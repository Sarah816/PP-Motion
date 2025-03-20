# python train.py --exp_name corrected_lossplcc_perpair_safe \
#     --dataset corrected \
#     --learning_rate 4e-5 --batch_size 32 --gpu_indices "6,7" --loss_type "plcc"\
#     --save_latest --lr_decay --big_model --enable_phys --save_checkpoint

# python train.py --exp_name corrected_lossplcc_perbatch_exp1 \
#     --dataset corrected \
#     --learning_rate 4e-5 --batch_size 32 --gpu_indices "6,7" --loss_type "plcc"\
#     --save_latest --lr_decay --big_model --enable_phys --save_checkpoint\
#     --load_model output/corrected_lossplcc_perbatch/best_checkpoint.pth

python train.py --exp_name corrected_lossmse_exp1 \
    --dataset corrected \
    --learning_rate 4e-5 --batch_size 32 --gpu_indices "6,7" --loss_type "mse"\
    --save_latest --lr_decay --big_model --enable_phys --save_checkpoint
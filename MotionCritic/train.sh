# tmux 4
# python train.py --exp_name lossplcc_test \
#     --dataset corrected \
#     --learning_rate 2e-5 --batch_size 64 --gpu_indices "4" --loss_type "plcc"\
#     --save_latest --lr_decay --big_model --enable_phys --save_checkpoint\
#     --phys_coef 0.3

# # tmux 1
# python train.py --exp_name norm_loccplcc_perbatch_phys0.3 \
#     --dataset corrected \
#     --learning_rate 2e-5 --batch_size 64 --gpu_indices "1,7" --loss_type "plcc"\
#     --save_latest --lr_decay --big_model --enable_phys --save_checkpoint\
#     --phys_coef 0.3



export CUDA_VISIBLE_DEVICES=0

# python -m render.render_batch\
#     --path ../save/finetuned/mdmft_physcritic_test1/step0 \
#     --exclude_gt --render_folder


python -m render.render_batch\
    --data_path data/motion_dataset/mlist_mdmval.pth \
    --output_path render_output/mdmval\
    --render_index 2745 2746 2747 1833 1834 1835 2112 2113 2114 \
    --exclude_gt

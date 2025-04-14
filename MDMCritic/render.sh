export CUDA_VISIBLE_DEVICES=0

python -m render.render_batch\
    --path ../data/val_dataset_for_metrics/mdm-fulleval.pth \
    --exclude_gt

export CUDA_VISIBLE_DEVICES=0

python -m render.render_batch\
    --folder ../save/finetuned/mdmft_physcritic_test1/step100 \
    --exclude_gt

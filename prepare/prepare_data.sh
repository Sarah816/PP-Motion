cd MotionCritic
mkdir -p data
cd data

echo -e "download physical annotation and save to data/phys_annotation\n"
gdown --folder https://drive.google.com/drive/folders/1rWq5GJEE_Cnkh3hvUZ51EBUyp3aaS06P

echo -e "download MotionPercept dataset and dave to data/motion_dataset\n"
gdown --folder https://drive.google.com/drive/folders/1A8x4o_xJxsVTVETJ2VspEjmg0wVtW29F

echo -e "download data-prompt mapping for per-prompt training and evaluation\n"
gdown --folder https://drive.google.com/drive/folders/1C4X7MpqmFYRsfGZtObj3jiRqhNNYAPFI

cd ..
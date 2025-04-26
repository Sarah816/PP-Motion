cd MotionCritic
mkdir -p data
cd data

echo -e "download ground truth mpjpe\n"
gdown --folder https://drive.google.com/drive/folders/1rWq5GJEE_Cnkh3hvUZ51EBUyp3aaS06P

echo -e "download mdm validation dataset\n"
gdown https://drive.google.com/uc?id=19eiE3WCON-b-y4FT0JJp7UZQc2iOqpyb

echo -e "download mdm training dataset\n"
gdown https://drive.google.com/uc?id=114ubH120XjNUAeQwhbBQ03HUg7pjkBO9

echo -e "download mdmval and flame dataset for evaluation\n"
gdown --folder https://drive.google.com/drive/folders/1nrpJsRY-pIBekwV9VHsQMkVIkPnLG6KO

echo -e "download data-prompt mapping for per-prompt training and evaluation\n"
gdown --folder https://drive.google.com/drive/folders/1C4X7MpqmFYRsfGZtObj3jiRqhNNYAPFI

cd ..
# PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation

This is the official PyTorch implementation of the paper "PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation" (ACM MM 2025).

## Dependencies

### Environment

Create new conda environment and install pytroch:

```
conda env create -f environment.yml
conda activate mocritic
```

Download and setup Isaac Gym.

### Dataset & Pretrained Model

Use the following script to download dataset and trained models.
```
bash prepare/prepare_data.sh
bash prepare/prepare_pretrained.sh
```

Download SMPL paramters from SMPL and SMPLX. Put them in the data/smpl folder, unzip them into 'data/smpl' folder. 

### Model

## Important files
```
PP-Motion/                   
├── MotionCritic/      
│   ├── train.sh
│   ├── metrics.sh
│   ├── render.sh
│   ├── data/
│       ├── mapping/                         # Data-prompt mapping, used for per-prompt training and evaluation
│           ├── mdm-fulleval_category.json
│           └── mdmtrain_category.json
│       ├── mpjpe/                           # GT mpjpe
│           ├── mdmtrain_mpjpe_corrected_norm.npy
│           ├── mdmval_mpjpe_norm.npy
│           └── flame_fulleval_mpjpe_norm.npy
│       ├── motion_dataset/
│           ├── mlist_flame_fulleval.pth
│           └── mlist_mdmval_fulleval.pth
│       ├── gt-packed/                       # Only used for metric calculation
│       ├── motion_dataset/mlist_mdmtrain_corrected.pth     # Train dataset
│       └── motion_dataset/mlist_mdmval.pth                 # Validation dataset, same as mlist_mdmval_fulleval.pth
│   ├── output/             # Checkpoints are saved here
│   ├── pretrained/         # Pretrained models
│       ├── motioncritic_pre.pth             # MotionCritic model
│       └── phys_pre.pth                     # PP-Motion model
│   └── stats/              # Metric statistics are saved here
└── MDMCritic/
```

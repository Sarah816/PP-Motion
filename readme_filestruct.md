# Important files
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
│           └── flame_fulleval_mpjpe.npy
│       ├── val_dataset_for_metrics/
│           ├── flame-fulleval.pth
│           └── mdmval-fulleval.pth
│       ├── gt-packed/                       # Only used for metric calculation
│       ├── mlist_mdmtrain_corrected.pth     # Train dataset
│       └── mlist_mdmval.pth                 # Validation dataset, same as mdmval-fulleval.pth
│   ├── output/             # Checkpoints are saved here
│   ├── pretrained/         # Pretrained models
│       ├── motioncritic_pre.pth             # MotionCritic model
│       └── phys_pre.pth                     # PP-Motion model
│   └── stats/              # Metric statistics are saved here
└── MDMCritic/
```

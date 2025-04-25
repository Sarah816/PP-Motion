```
MotionCritic/                   
├── MotionCritic/      
│   ├── train.sh
│   ├── metrics.sh
│   ├── render.sh
│   ├── data/
│       ├── mapping/
│           ├── mdm-fulleval_category.json
│           └── mdmtrain_category.json
│       ├── mpjpe/
│           ├── mdmtrain_mpjpe_corrected_norm.npy
│           ├── mdmval_mpjpe_norm.npy
│           └── flame_fulleval_mpjpe.npy
│       ├── val_dataset_for_metrics/
│           ├── flame-fulleval.pth
│           └── mdmval-fulleval.pth
│       ├── gt-packed/                       # Only used for metric calculation
│       ├── mlist_mdmtrain_corrected.pth     # Train dataset
│       └── mlist_mdmval.pth                 # Validation dataset, same as mdmval-fulleval.pth
│   ├── output/             # Checkpoints
│   └── stats/              # Metric statistics are saved here
└── MDMCritic/
```

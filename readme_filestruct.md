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

# PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation

This is the official PyTorch implementation of the paper "PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation" (ACM MM 2025).

## Dependencies

### Environment

Create new conda environment and install pytroch:

```
conda env create -f environment.yml
conda activate mocritic
```

<!-- Download and setup Isaac Gym. -->

### Dataset & Pretrained Model

Use the following script to download dataset and pretrained models.
```
bash prepare/prepare_data.sh
bash prepare/prepare_pretrained.sh
```

Use the following script to download SMPL parameters.
```
bash prepare/prepare_smpl.sh
```

## Important files
```
PP-Motion/                   
├── MotionCritic/      
│   ├── train.sh
│   ├── metrics.sh
│   ├── render.sh
│   ├── data/
│       ├── mapping/                         # Data-prompt mapping, used for per-prompt training and evaluation
│           ├── mdmval_category.json
│           └── mdmtrain_category.json
│       ├── phys_annotation/                 # Physical annotations
│           ├── mdmtrain_mpjpe_norm.npy
│           ├── mdmval_mpjpe_norm.npy
│           └── flame_mpjpe_norm.npy
│       ├── motion_dataset/
│           ├── mlist_flame.pth
│           ├── mlist_mdmtrain.pth
│           └── mlist_mdmval.pth
│       └── gt-packed/                       # Only used for metric calculation
│   ├── output/             # Checkpoints and pretrained models are saved here
│       ├── motion-critic_pretrained
│           └── checkpoint_latest.pth        # MotionCritic model
│       └── pp-motion_pretrained
│           └── checkpoint_latest.pth        # PP-Motion model (Ours)
│   └── stats/              # Metric statistics are saved here
└── MDMCritic/
```

## Evaluating PP-Motion Performance

This script runs motion quality evaluation for both **Ours (PP-Motion)** and **MotionCritic** (baseline) models.

1. **Prepare checkpoints**

   Place your model checkpoint in the directory: `output/{exp_name}/checkpoint_latest.pth`

Example:
```
    
output/pp-motion_pretrained/checkpoint_latest.pth     # Checkpoint for PP-Motion model
output/motion-critic_pretrained/checkpoint_latest.pth # Checkpoint for MotionCritic (baseline) model

```

2. **Run evaluation**

```bash
cd MotionCritic
bash metrics.sh
```

or run manually:

```bash
    python metric/metrics.py \
    --mode mdmval \
    --exp_name pp-motion_pretrained \
    --checkpoint checkpoint_latest \
    # Optional flags:
    # --mode                 # Validation dataset, options: [mdmval, flame]
    # --calc_baseline_metric # Evaluate on MotionCritic (baseline) metric
    # --calc_phys_metric     # Evaluate on physical plausibility metrics
    # --calc_gt_metric       # Evaluate on GT-based metrics (Joint/Root AE/AVE)
    # --calc_perprompt       # Compute per-prompt correlations
```

3. **Results**
    - Metric evaluation results are saved in: `stats/metric_results_{val_dataset}_{exp_name}.json`
    - Intermediate results (metric output scores) are saved in: `stats/scores/{exp_name}/score_{val_dataset}_{checkpoint}.npy`

## Training PP-Motion

```bash
cd MotionCritic
bash train.sh
```
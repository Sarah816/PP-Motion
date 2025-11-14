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
│           ├── visexample.pth
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

## Quick Demo
Quickly get PP-Motion score for a motion sequence and render the motion:

```
cd MotionCritic
python visexample.py
```

Or run manually:

```python
from lib.model.load_critic import load_critic
from render.render import render_multi
import torch
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
render_output_path = "render_output"
motion_pth = "data/visexample.pth" # torch tensor: [bs, frame, 25, 3]), axis-angle with 24 SMPL joints and 1 XYZ root location
motion_seq = torch.load(motion_pth, map_location=device)
critic_model = load_critic("output/pp-motion_pretrained/checkpoint_latest.pth", device)
critic_scores = critic_model.module.batch_critic(motion_seq).tolist()
comments = []
output_paths = []
print(f"critic scores are {critic_scores}")
for idx, score in enumerate(critic_scores):
    score = round(score[0], 2)
    comments.append(f"PP-Motion score is: {score}")
    output_paths.append(os.path.join(render_output_path, f"visexample_{idx}.mp4"))
# rendering
print("Rendering...")
motion_seq = motion_seq.permute(0, 2, 3, 1) # [batch_size, 25, 3, num_frames=60]
render_multi(motion_seq, device, comments, output_paths, pose_format="rotvec")
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

Or run manually:

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
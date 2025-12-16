# PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation

This is the official PyTorch implementation of the paper "PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation" (ACM MM 2025).

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![arXiv](https://img.shields.io/badge/arXiv-2508.08179-b31b1b.svg)](https://arxiv.org/abs/2508.08179) <a href="https://sarah816.github.io/pp-motion-site/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>

![framework](./assets/pipeline.jpg)

## Dependencies

### Environment

Create new conda environment and install pytroch:

```
conda env create -f environment.yml
conda activate ppmotion
```

<!-- Download and setup Isaac Gym. -->

### Dataset & Pretrained Model

Use the following script to download dataset and pretrained models.
```
cd PP-Motion
bash prepare/prepare_data.sh
bash prepare/prepare_pretrained.sh
```

Use the following script to download SMPL parameters.
```
bash prepare/prepare_smpl.sh
```

### Important files
```
PP-Motion/                   
├── PP-Motion/      
│   ├── train.sh
│   ├── metrics.sh
│   ├── render.sh
│   ├── data/
│       ├── mapping/                     # Data-prompt mapping, used for per-prompt training and evaluation
│           ├── mdmval_category.json
│           └── mdmtrain_category.json
│       ├── phys_annotation/             # Physical annotations
│           ├── mdmtrain_mpjpe_norm.npy
│           ├── mdmval_mpjpe_norm.npy
│           └── flame_mpjpe_norm.npy
│       ├── human_annotation/            # Human perception annotations
│           ├── flame-fulleval.json
│           └── mdm-fulleval.json
│       ├── motion_dataset/
│           ├── visexample.pth
│           ├── mlist_flame.pth
│           ├── mlist_mdmtrain.pth
│           └── mlist_mdmval.pth
│       └── gt-packed/                   # Optional, only used for gt-based metric calculation
│   ├── output/    # Checkpoints and pretrained models are saved here
│       ├── motion-critic_pretrained
│           └── checkpoint_latest.pth    # MotionCritic model
│       └── pp-motion_pretrained
│           └── checkpoint_latest.pth    # PP-Motion model (Ours)
│   └── stats/     # Metric statistics are saved here
└── MDMCritic/
```


## Quick Demo

### Using PP-Motion as Metric
Quickly get PP-Motion score for a motion sequence and render the motion:

```
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

### Rendering
If you only want to render a motion sequence from a dataset:

```
bash render.sh
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

If you add `--calc_gt_metric`, you need to first download gt data:

```
bash prepare/prepare_gt.sh
```

3. **Results**
    - Metric evaluation results are saved in: `stats/metric_results_{val_dataset}_{exp_name}.json`
    - Intermediate results (metric output scores) are saved in: `stats/scores/{exp_name}/score_{val_dataset}_{checkpoint}.npy`

## Training PP-Motion

```bash
bash train.sh
```

Or run manually:

```
python train.py --exp_name pp-motion \
    --big_model --learning_rate 2e-5 --batch_size 64  --lr_decay \
    --save_latest --save_checkpoint \
    --loss_type "plcc" --enable_phys --phys_coef 0.2 \
```

This training script use `PP-Motion/data/motion_dataset/mlist_mdmtrain.pth` as training data, and `PP-Motion/data/motion_dataset/mlist_mdmval.pth` as evaluation data.


## Generating Physical Annotation

Details about generating physical annotations can be found in `PP-Annotation` folder. Run the script below from the project root directory:

```
git submodule update --init --recursive
cd PP-Annotation
```

## More Information

### Dataset Documentation

- [Dataset](docs/dataset.md)
- [Motion files](docs/motion.md)

<!-- ### Trouble Shooting -->


## Citation
If you find our work useful for your project, please consider citing the paper:
```bibtex
@inproceedings{zhao2025pp,
  title={PP-Motion: Physical-Perceptual Fidelity Evaluation for Human Motion Generation},
  author={Zhao, Sihan and Wang, Zixuan and Luan, Tianyu and Jia, Jia and Zhu, Wentao and Luo, Jiebo and Yuan, Junsong and Xi, Nan},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={6840--6849},
  year={2025}
}
```


## Acknowledgement

This repository is built on top of: 
* PP-Motion metric framework is from: [MotionCritic](https://github.com/ou524u/MotionCritic)
* Physics annotation generation framework is from: [PHC](https://github.com/ZhengyiLuo/PHC)
* Physics simulator is from: [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)


If you use PP-Motion in your work, please also cite the original datasets and methods on which our work is based.

```bibtex
@inproceedings{motioncritic2025,
    title={Aligning Motion Generation with Human Perceptions},
    author={Wang, Haoru and Zhu, Wentao and Miao, Luyi and Xu, Yishu and Gao, Feng and Tian, Qi and Wang, Yizhou},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2025}
}
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}
@inproceedings{
  tevet2023human,
  title={Human Motion Diffusion Model},
  author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023}
}
@inproceedings{guo2020action2motion,
  title={Action2motion: Conditioned generation of 3d human motions},
  author={Guo, Chuan and Zuo, Xinxin and Wang, Sen and Zou, Shihao and Sun, Qingyao and Deng, Annan and Gong, Minglun and Cheng, Li},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2021--2029},
  year={2020}
}
@inproceedings{kim2023flame,
  title={Flame: Free-form language-based motion synthesis \& editing},
  author={Kim, Jihoon and Kim, Jiseob and Choi, Sungjoon},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={7},
  pages={8255--8263},
  year={2023}
}
@inproceedings{ji2018large,
  title={A large-scale RGB-D database for arbitrary-view human action recognition},
  author={Ji, Yanli and Xu, Feixiang and Yang, Yang and Shen, Fumin and Shen, Heng Tao and Zheng, Wei-Shi},
  booktitle={Proceedings of the 26th ACM international Conference on Multimedia},
  pages={1510--1518},
  year={2018}
}
@incollection{loper2023smpl,
  title={SMPL: A skinned multi-person linear model},
  author={Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J},
  booktitle={Seminal Graphics Papers: Pushing the Boundaries, Volume 2},
  pages={851--866},
  year={2023}
}
```

'''
数据处理：1. 统一坐标系（y-up to z-up） 2. 统一帧率（24fps to 60fps）
'''
from lib.model.load_critic import load_critic
import torch
import joblib
from parsedata import transform_dataset, into_critic
from render.render import render_multi
import numpy as np
from tqdm import *
import os
import json
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
处理轴角表示或rotation matrix的MDM生成数据（数据已转化为MotionCritic的轴角表示格式）
'''

exp_name = "mdmft_physcritic_test1"
step = 100
dataset_path = f"MDMCritic/save/finetuned/{exp_name}/step{step}/step{step}-evalbatch.pth"
output_dir = f"MotionCritic/data/mdm_gen/{exp_name}-step{step}.pth"
data_raw = torch.load(dataset_path) # [batch_size, 25, 6, 60]
scores = data_raw['score']
print(f"mean score: {sum(scores) / len(scores)}")
motion_raw = into_critic(data_raw['motion']) # [batch_size, 60, 25, 3]
data_len = len(motion_raw)
data_processed = []
# random_idx = random.sample(range(0, data_len), 120)
# with open("MotionCritic/data/finetune_gen/random_index_test1.json", 'w') as f:
#     json.dump(random_idx, f)
for i in tqdm(range(data_len)):
    motion = motion_raw[i].cpu()
    assert(motion.shape == (60, 25, 3))
    data_processed.append(transform_dataset(motion))
data_processed = torch.stack(data_processed) # [batch_size, 150, 25, 3]
print(f'Saving data to {output_dir}, shape {data_processed.shape}')
torch.save(data_processed, output_dir)
exit(0)

'''处理MotionCritic原始数据'''

dataset_path = "MotionCritic/data/finetune_gen/phys-motions-step0100.pth"
output_dir = f"{dataset_path}_60fps.pth"
motion_raw = torch.load(dataset_path)
data_len = len(motion_raw)
data_better = []
data_worse = []
# mdm_val_small = []
# len(data_raw)
for i in tqdm(range(0, len(motion_raw))):
    motion_better = motion_raw[i]['motion_better'] # [60, 25, 3]
    motion_worse = motion_raw[i]['motion_worse']
    assert(motion_better.shape == (60, 25, 3))
    data_better.append(transform_dataset(motion_better))
    data_worse.append(transform_dataset(motion_worse))
# torch.save(data_better, "MotionCritic/data/flame_better_60fps.pth")
# torch.save(data_worse, "MotionCritic/data/flame_worse_60fps.pth")
data_processed = torch.stack((torch.stack(data_better), torch.stack(data_worse))) # [2, batch_size, 150, 25, 3]
print(f'Saving data to {output_dir}, shape {data_processed.shape}')
torch.save(data_processed, output_dir)
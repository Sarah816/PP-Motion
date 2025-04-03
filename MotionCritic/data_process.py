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
from lib.utils.rotation2xyz import Rotation2xyz

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda:0')


def visualize_vertices(vertices, title):
    '''
    Input vertices: shape [num_frames, 6890, 3]
    Only visualize the first frame
    Save pic to render_output/pointcloud/{title}.png
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vertices = vertices[0].cpu().numpy() # Only visualize the first frame
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    fig = plt.figure(dpi = 100)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, color="blue")  # 颜色用 z 值表示
    ax.view_init(elev=10., azim=-40)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.5, -0.5)
    ax.set_zlim(0, 1.5)
    plt.savefig(f"render_output/pointcloud/{title}.png", dpi=300)
    plt.close()
    # exit(0)

@torch.no_grad()
def fix_height_smpl_critic(motion):
    # no filtering, just fix height
    # motion: [num_frames=60, 25, 3]
    motion = motion.permute(1, 2, 0).unsqueeze(0) # motion: [1, 25, 3, num_frames=60]
    motion = motion.to(device)
    rot2xyz = Rotation2xyz(device=device)
    # 只获取运动第一帧的vertices
    vertices = rot2xyz(motion[:, :, :, 0:1], mask=None,
                       pose_rep='rotvec', translation=True, glob=True,
                       jointstype='vertices', betas=None, beta=0, glob_rot=None,
                       vertstrans=True) # [1, 6890, 3, num_frames=1]
    vertices = vertices.squeeze(dim=0).permute(2, 0, 1) # [num_frames=1, 6890, 3]
    # 获得所有顶点的y坐标的最小值
    gp = torch.min(vertices[:, :, 1])
    # motion每一帧的y坐标都减去gp，即第一帧的最低顶点落在地上
    motion[:, -1, 1] -= gp
    # vertices[:, :, 1] -= gp
    # visualize_vertices(vertices, "test1-fixheight-y-up")
    # vertices[:, :, [1, 2]] = vertices[:, :, [2, 1]]
    # visualize_vertices(vertices, "test1-fixheight-z-up")
    motion = motion.squeeze(dim=0).permute(2, 0, 1)
    return motion.cpu() # shape: [num_frames=60, 25, 3]


'''
处理轴角表示的MotionCritic数据：fix height
'''
# dataset_path = "data/val_dataset_for_metrics/mdm-fulleval.pth"
dataset_path = "data/mlist_mdmtrain_corrected.pth"
motion_raw = torch.load(dataset_path)
motion_fixheight = []
for i in tqdm(range(len(motion_raw))):
    motion_better = motion_raw[i]['motion_better'] # [60, 25, 3]
    motion_better_fix = fix_height_smpl_critic(motion_better)
    motion_worse = motion_raw[i]['motion_worse']
    motion_worse_fix = fix_height_smpl_critic(motion_worse)
    motion_fixheight.append({"motion_better": motion_better_fix, "motion_worse": motion_worse_fix})
torch.save(motion_fixheight, dataset_path.strip(".pth")+"-fixheight.pth")
exit(0)
    

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
for i in tqdm(range(len(motion_raw))):
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
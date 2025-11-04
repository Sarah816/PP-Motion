import sys
import os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

import torch
import numpy as np
import json
import torch.nn.functional as F
from sklearn.metrics import log_loss
from lib.utils.rotation2xyz import Rotation2xyz
import argparse
import pandas as pd
from tqdm import tqdm

from correlation import metric_correlation
from parsedata import into_critic
from critic_score import get_val_scores
from uhc.smpllib.smpl_eval import compute_phys_metrics


np.set_printoptions(precision=2, suppress=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Motion Critic Evaluation')
    parser.add_argument('--mode', type=str, choices=['mdmval', 'flame'], 
                       default='mdmval',
                       required=False,
                       help='Evaluation mode: mdm or flame')
    parser.add_argument('--exp_name', type=str, 
                       required=True,)
    parser.add_argument('--checkpoint', type=str, 
                       required=False,
                       default='checkpoint_latest',)
    parser.add_argument('--device_id', type=str, 
                       required=False,
                       default='0',)
    parser.add_argument('--calc_perprompt', action='store_true',
                        help='calculate metrics per prompt')
    parser.add_argument('--calc_baseline_metric', action='store_true',
                        help='whether to calculate baseline metric: MotionCritic')
    parser.add_argument('--calc_phys_metric', action='store_true',
                        help='whether to calculate physical metrics: penetration, skating, floating, PHC')
    parser.add_argument('--calc_gt_metric', action='store_true',
                        help='whether to calculate GT-based metrics: Joint AE/AVE, Root AE/AVE')
    return parser.parse_args()

# gt_humanact12 = [torch.load(os.path.join(PROJ_DIR, f'data/gt-packed/gt-humanact12/motion-gt{i}.pth'))['motion'] for i in range(12)]
# gt_uestc = [torch.load(os.path.join(PROJ_DIR, f'data/gt-packed/gt-uestc/motion-gtuestc{i}.pth'))['motion'] for i in range(40)]
# gt_flame = [torch.load(os.path.join(PROJ_DIR, f'data/gt-packed/gt-flame/motion-gtflame.pth'))]

# gt_humanact12xyz = torch.load(os.path.join(PROJ_DIR, f'data/gt-packed/humanact12-gt-jointxyz.pt'))
# gt_uestcxyz = torch.load(os.path.join(PROJ_DIR, f'data/gt-packed/uestc-gt-jointxyz.pt'))
# gt_flamexyz = torch.load(os.path.join(PROJ_DIR, f'data/gt-packed/flame-gt-jointxyz.pt'))


gt_humanact12 = None
gt_uestc = None
gt_flame = None

gt_humanact12xyz = None
gt_uestcxyz = None
gt_flamexyz = None


def load_gt_datasets(mode):
    global gt_humanact12, gt_uestc, gt_flame
    global gt_humanact12xyz, gt_uestcxyz, gt_flamexyz

    if mode == 'mdmval':
        # rotation/pose gt
        if gt_humanact12 is None:
            gt_humanact12 = [
                torch.load(os.path.join(PROJ_DIR, 'data/gt-packed/gt-humanact12', f'motion-gt{i}.pth'))['motion']
                for i in range(12)
            ]
        if gt_uestc is None:
            gt_uestc = [
                torch.load(os.path.join(PROJ_DIR, 'data/gt-packed/gt-uestc', f'motion-gtuestc{i}.pth'))['motion']
                for i in range(40)
            ]
        # xyz gt
        if gt_humanact12xyz is None:
            gt_humanact12xyz = torch.load(os.path.join(PROJ_DIR, 'data/gt-packed/humanact12-gt-jointxyz.pt'))
        if gt_uestcxyz is None:
            gt_uestcxyz = torch.load(os.path.join(PROJ_DIR, 'data/gt-packed/uestc-gt-jointxyz.pt'))

    elif mode == 'flame':
        if gt_flame is None:
            gt_flame = [
                torch.load(os.path.join(PROJ_DIR, 'data/gt-packed/gt-flame', 'motion-gtflame.pth'))
            ]
        if gt_flamexyz is None:
            gt_flamexyz = torch.load(os.path.join(PROJ_DIR, 'data/gt-packed/flame-gt-jointxyz.pt'))



def visualize_vertices(vertices, title):
    '''
    vertices shape: [num_frames, 6890, 3]
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vertices = vertices[0].cpu().numpy()
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
    ax.set_zlim(-0.5, 0.5)

    plt.savefig(f"render_output/pointcloud/{title}.png", dpi=300)
    plt.close()
    # exit(0)

def get_vertices(motion, fix_height=True):
    '''
    return vertices in z-up coordinate system
    '''
    # motion: [batch_size, num_frames=60, 25, 3]
    # motion: [batch_size, 25, 3, num_frames=60]
    motion = motion.permute(0, 2, 3, 1) # motion: [batch_size, 25, 3, num_frames=60]
    motion = motion.to(device)
    rot2xyz = Rotation2xyz(device=device)
    vertices = rot2xyz(motion, mask=None,
                       pose_rep='rotvec', translation=True, glob=True,
                       jointstype='vertices', betas=None, beta=0, glob_rot=None,
                       vertstrans=True)
    v = vertices.clone() # [batch_size, 6890, 3, num_frames=60]
    v = v.permute(0, 3, 1, 2) # [batch_size, num_frames=60, 6890, 3]
    v[:, :, :, [1, 2]] = v[:, :, :, [2, 1]] # 交换y,z轴数据
    # Fix height
    if fix_height:
        for i in range(v.shape[0]):
            gp = torch.min(v[i, 0, :, 2]) # 让每条motion的第0帧的最低点落地
            v[i, :, :, 2] -= gp
    # visualize_vertices(v[0], "vertices_z-up")
    # exit(0)
    return v

def compute_phys(dataset_pth):
    dataset = torch.load(dataset_pth)
    compute_pen = []
    compute_skate = []
    compute_float = []
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        motion_better = item['motion_better']
        motion_worse = item['motion_worse']
        motion = torch.stack((motion_better, motion_worse), dim=0)
        vertices = get_vertices(motion)
        metric_better = compute_phys_metrics(vertices[0])
        metric_worse = compute_phys_metrics(vertices[1])
        compute_pen.append(np.array([metric_better["penetration"], metric_worse["penetration"]]))
        compute_skate.append(np.array([metric_better["skate"], metric_worse["skate"]]))
        compute_float.append(np.array([metric_better["float"], metric_worse["float"]]))
    compute_pen = np.stack(compute_pen, axis=0)
    compute_skate = np.stack(compute_skate, axis=0)
    compute_float = np.stack(compute_float, axis=0)
    # print("penetration:", compute_pen)
    # print("skating:", compute_skate)
    # print("floating:", compute_float)
    return compute_pen, compute_skate, compute_float


# Remove global evalmdm and evalflame declarations
# Instead, use args.mode throughout the code

def choose_gt_dataset_from_filename(file_name):
    if args.mode == 'mdmval':
        action_class = extract_number_from_filename(file_name)
        if file_name[3] == 'a':
            return gt_humanact12[action_class]
        elif file_name[3] == 'u':
            return gt_uestc[action_class]
    
    if args.mode == 'flame':
        return gt_flame[0]


def extract_number_from_filename(file_name):
    first_dash_index = file_name.find('-')
    second_dash_index = file_name.find('-', first_dash_index + 1)
    third_dash_index = file_name.find('-', second_dash_index + 1)
    if second_dash_index == -1 or third_dash_index == -1:
        return None
    number_str = file_name[second_dash_index + 1:third_dash_index]
    try:
        number = int(number_str)
        return number
    except ValueError:
        return None


def choose_gtxyz_dataset_from_filename(file_name):
    if args.mode == 'mdmval':
        action_class = extract_number_from_filename(file_name)

        if file_name[3] == 'a':
            return gt_humanact12xyz[action_class]
        elif file_name[3] == 'u':
            return gt_uestcxyz[action_class]
    
    if args.mode == 'flame':
        return gt_flamexyz[0]


def build_gt_xyz():
    device = 'cpu'
    rot2xyz = Rotation2xyz(device=device)
    if args.mode == 'mdmval':
        global gt_humanact12xyz
        global gt_uestcxyz
        for gt in tqdm(gt_humanact12):
            # gt shape: [batch_size, 25, 6, num_frames=60]
            gt_xyz = rot2xyz(gt, mask=None,
                            pose_rep='rot6d', translation=True, glob=True,
                            jointstype='smpl', betas=None, beta=0, glob_rot=None,
                            vertstrans=True)
            # shape is [batch_size, 24, 3, 60]
            gt_xyz = gt_xyz.permute(0, 3, 1, 2) # shape is [batch_size, 60, 24, 3]
            gt_humanact12xyz.append(gt_xyz)
        
        # gt_humanact12xyz = torch.stack(gt_humanact12xyz)
        # print(f"gt_humanact12xyz shape {gt_humanact12xyz.shape}")
        torch.save(gt_humanact12xyz, os.path.join(PROJ_DIR, f'data/gt-packed/humanact12-gt-jointxyz.pt'))
        
        for gt in tqdm(gt_uestc):
            gt_xyz = rot2xyz(gt, mask=None,
                            pose_rep='rot6d', translation=True, glob=True,
                            jointstype='smpl', betas=None, beta=0, glob_rot=None,
                            vertstrans=True)
            # shape is [batch_size, 24, 3, 60]
            gt_xyz = gt_xyz.permute(0, 3, 1, 2)
            gt_uestcxyz.append(gt_xyz)
        
        # gt_uestcxyz = torch.stack(gt_uestcxyz)
        # print(f"gt_uestcxyz shape {gt_uestcxyz.shape}")
        torch.save(gt_uestcxyz, os.path.join(PROJ_DIR, f'data/gt-packed/uestc-gt-jointxyz.pt'))
        

    if args.mode == 'flame':
        global gt_flamexyz
        for gt in tqdm(gt_flame):
            gt_xyz = rot2xyz(gt, mask=None,
                            pose_rep='rot6d', translation=True, glob=True,
                            jointstype='smpl', betas=None, beta=0, glob_rot=None,
                            vertstrans=True)
            # shape is [batch_size, 24, 3, 60]
            gt_xyz = gt_xyz.permute(0, 3, 1, 2)
            gt_flamexyz.append(gt_xyz)
        
        gt_flamexyz = torch.stack(gt_flamexyz)
        print(f"gt_flamexyz shape {gt_flamexyz.shape}") # [1, batch_size, 60, 24, 3]
        torch.save(gt_flamexyz, os.path.join(PROJ_DIR, f'data/gt-packed/flame-gt-jointxyz.pth'))



def compute_AE(batch_vectors, ground_truth_vectors):
    # batch: [batch_size, 60, 1, 3]
    # gt: [gt_batch_size, 60, 1, 3]
    # print(f"tensor shapes: {batch_vectors.shape}, {ground_truth_vectors.shape}")
    l2_losses = torch.sqrt(torch.sum((batch_vectors.unsqueeze(1) - ground_truth_vectors.unsqueeze(0)) ** 2, dim=-1))
    mean_l2_losses = torch.mean(l2_losses, dim=(1, 2, 3))
    return mean_l2_losses
    

def compute_AVE(batch_vectors, ground_truth_vectors):
    # batch: [batch_size, 60, 1, 3]
    # gt: [gt_batch_size, 60, 1, 3]
    batch_variances = torch.var(batch_vectors, dim=1, keepdim=True)  # [batch_size, 1, 1, 3]
    gt_variances = torch.var(ground_truth_vectors, dim=1, keepdim=True)  # [gt_batch_size, 1, 1, 3]

    l2_losses = torch.sqrt(torch.sum((batch_variances.unsqueeze(1) - gt_variances.unsqueeze(0)) ** 2, dim=-1))
    mean_l2_losses = torch.mean(l2_losses, dim=(1, 2, 3))
    
    return mean_l2_losses


def compute_PFC(batch_vectors):
    # [batchsize, 60, 22, 3]
    delta_t = 1/30
    scores = []
    for batch_vec in  batch_vectors:
        root_v = (batch_vec[1:,0,:] - batch_vec[:-1,0,:])/delta_t
        root_a = (root_v[1:] - root_v[:-1])/delta_t
        root_a = np.linalg.norm(root_a, axis=-1)
        scaling = root_a.max()
        root_a /= scaling


        foot_idx = [7, 10, 8, 11]
        flat_dirs = [0, 2]
        feet = batch_vec[:,foot_idx]
        foot_v = np.linalg.norm(
                feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
            )  
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        # print(f"shape foot_means {foot_mins.shape}, shape root_a {root_a.shape}")
        foot_loss = (
                foot_mins[:, 0] * foot_mins[:, 1] * root_a
            )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        # names.append(pkl)
        # accelerations.append(foot_mins[:, 0].mean())
    scores_tensor = [torch.tensor(score) for score in scores]
    scores_tensor = torch.stack(scores_tensor)
    return scores_tensor


def rootloc_pairs_from_filename(file_name, choise):
    better_loc = []
    worse_loc = []
    npz_file = np.load(os.path.join(motion_location, file_name), allow_pickle=True)
    motion = npz_file['arr_0'].item()['motion'] # shape:[batch_size,25,6,60]
    motion = np.array(motion).transpose(0, 3, 1, 2) # shape:[batch_size,60,25,6]


    
    root_loc = torch.from_numpy(motion[:,:,24:25,0:3]) # shape:[batch_size,60,1,3], batch_size=1
    # print(f"file {file_name}, motion shape {type(root_loc)} {root_loc.shape}, choice {choise}")

    if args.mode == 'mdmval':
        if choise == 'A':
            better_loc = [root_loc[0], root_loc[0], root_loc[0]]
            worse_loc = [root_loc[1], root_loc[2], root_loc[3]]
        elif choise == 'B':
            better_loc = [root_loc[1], root_loc[1], root_loc[1]]
            worse_loc = [root_loc[0], root_loc[2], root_loc[3]]
        elif choise == 'C':
            better_loc = [root_loc[2], root_loc[2], root_loc[2]]
            worse_loc = [root_loc[0], root_loc[1], root_loc[3]]
        elif choise == 'D':
            better_loc = [root_loc[3], root_loc[3], root_loc[3]]
            worse_loc = [root_loc[0], root_loc[1], root_loc[2]]
    
    if args.mode == 'flame':
        if choise == 'A':
            worse_loc = [root_loc[0], root_loc[0], root_loc[0]]
            better_loc = [root_loc[1], root_loc[2], root_loc[3]]
        elif choise == 'B':
            worse_loc = [root_loc[1], root_loc[1], root_loc[1]]
            better_loc = [root_loc[0], root_loc[2], root_loc[3]]
        elif choise == 'C':
            worse_loc = [root_loc[2], root_loc[2], root_loc[2]]
            better_loc = [root_loc[0], root_loc[1], root_loc[3]]
        elif choise == 'D':
            worse_loc = [root_loc[3], root_loc[3], root_loc[3]]
            better_loc = [root_loc[0], root_loc[1], root_loc[2]]

    
    better_loc = torch.stack(better_loc, dim=0)
    worse_loc = torch.stack(worse_loc, dim=0)

    return better_loc, worse_loc


def make_pairs(joints_xyz, choise):
    # data.shape[0] is 4 (4 data in a batch)
    
    if args.mode == 'mdmval':
        if choise == 'A':
            better_xyz = [joints_xyz[0], joints_xyz[0], joints_xyz[0]]
            worse_xyz = [joints_xyz[1], joints_xyz[2], joints_xyz[3]]
        elif choise == 'B':
            better_xyz = [joints_xyz[1], joints_xyz[1], joints_xyz[1]]
            worse_xyz = [joints_xyz[0], joints_xyz[2], joints_xyz[3]]
        elif choise == 'C':
            better_xyz = [joints_xyz[2], joints_xyz[2], joints_xyz[2]]
            worse_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[3]]
        elif choise == 'D':
            better_xyz = [joints_xyz[3], joints_xyz[3], joints_xyz[3]]
            worse_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[2]]

    if args.mode == 'flame':
        if choise == 'A':
            worse_xyz = [joints_xyz[0], joints_xyz[0], joints_xyz[0]]
            better_xyz = [joints_xyz[1], joints_xyz[2], joints_xyz[3]]
        elif choise == 'B':
            worse_xyz = [joints_xyz[1], joints_xyz[1], joints_xyz[1]]
            better_xyz = [joints_xyz[0], joints_xyz[2], joints_xyz[3]]
        elif choise == 'C':
            worse_xyz = [joints_xyz[2], joints_xyz[2], joints_xyz[2]]
            better_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[3]]
        elif choise == 'D':
            worse_xyz = [joints_xyz[3], joints_xyz[3], joints_xyz[3]]
            better_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[2]]

    
    better_xyz = torch.stack(better_xyz, dim=0)
    worse_xyz = torch.stack(worse_xyz, dim=0)

    return better_xyz, worse_xyz


def critic_data_from_filename(file_name, choise):
    npz_file = np.load(os.path.join(motion_location, file_name), allow_pickle=True)
    prompt = npz_file['arr_0'].item()['prompt']
    with open("data/mapping/flame_fulleval_prompt.txt", 'a') as f:
        f.write(f"{prompt}\n{prompt}\n{prompt}\n")
    motion = npz_file['arr_0'].item()['motion'] # shape:[4,25,6,60]
    motion = torch.from_numpy(np.array(motion))
    motion_critic = into_critic(motion) # shape:[4,60,25,3]
    better, worse = make_pairs(motion_critic, choise)  # shape:[3,60,25,3]
    return better, worse


def jointxyz_pairs_from_filename(file_name, choise):
    # Loading from: MotionCritic/datasets
    npz_file = np.load(os.path.join(motion_location, file_name), allow_pickle=True)
    motion = npz_file['arr_0'].item()['motion'] # shape:[batch_size,25,6,60]

    motion = torch.from_numpy(np.array(motion))
    device = 'cpu'
    rot2xyz = Rotation2xyz(device=device)

    joints_xyz = rot2xyz(motion, mask=None,
                       pose_rep='rot6d', translation=True, glob=True,
                       jointstype='smpl', betas=None, beta=0, glob_rot=None,
                       vertstrans=True)
    # print(f"joints_xyz shape {joints_xyz.shape}") # is [batch_size, 24, 3, 60]
    joints_xyz = joints_xyz.permute(0, 3, 1, 2) # [4, 60, 24, 3]

    if args.mode == 'mdmval':
        if choise == 'A':
            better_xyz = [joints_xyz[0], joints_xyz[0], joints_xyz[0]]
            worse_xyz = [joints_xyz[1], joints_xyz[2], joints_xyz[3]]
        elif choise == 'B':
            better_xyz = [joints_xyz[1], joints_xyz[1], joints_xyz[1]]
            worse_xyz = [joints_xyz[0], joints_xyz[2], joints_xyz[3]]
        elif choise == 'C':
            better_xyz = [joints_xyz[2], joints_xyz[2], joints_xyz[2]]
            worse_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[3]]
        elif choise == 'D':
            better_xyz = [joints_xyz[3], joints_xyz[3], joints_xyz[3]]
            worse_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[2]]

    if args.mode == 'flame':
        if choise == 'A':
            worse_xyz = [joints_xyz[0], joints_xyz[0], joints_xyz[0]]
            better_xyz = [joints_xyz[1], joints_xyz[2], joints_xyz[3]]
        elif choise == 'B':
            worse_xyz = [joints_xyz[1], joints_xyz[1], joints_xyz[1]]
            better_xyz = [joints_xyz[0], joints_xyz[2], joints_xyz[3]]
        elif choise == 'C':
            worse_xyz = [joints_xyz[2], joints_xyz[2], joints_xyz[2]]
            better_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[3]]
        elif choise == 'D':
            worse_xyz = [joints_xyz[3], joints_xyz[3], joints_xyz[3]]
            better_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[2]]

    
    better_xyz = torch.stack(better_xyz, dim=0)
    worse_xyz = torch.stack(worse_xyz, dim=0)

    return better_xyz, worse_xyz


def results_from_filename(file_name, choise, metric):
    
    gt = choose_gt_dataset_from_filename(file_name) # gt shape: [batch_size, 60, 25, 3], axis-angle
    gt_loc = gt.permute(0,3,1,2)[:,:,24:25,0:3]

    if metric == 'Root AE':
        better, worse = rootloc_pairs_from_filename(file_name, choise)
        gt_loc = gt.permute(0,3,1,2)[:,:,24:25,0:3]
        better_AE = compute_AE(better, gt_loc)
        worse_AE = compute_AE(worse, gt_loc)
        return better_AE, worse_AE
 
    elif metric == 'Root AVE':
        better, worse = rootloc_pairs_from_filename(file_name, choise)
        gt_loc = gt.permute(0,3,1,2)[:,:,24:25,0:3]
        better_AVE = compute_AVE(better, gt_loc)
        worse_AVE = compute_AVE(worse, gt_loc)
        return better_AVE, worse_AVE
    
    elif metric == 'Joint AE':
        better, worse = jointxyz_pairs_from_filename(file_name, choise)
        gt_xyz = choose_gtxyz_dataset_from_filename(file_name)
        better = better.to(device)
        worse = worse.to(device)
        gt_xyz = gt_xyz.to(device)
        better_AE = compute_AE(better, gt_xyz)
        worse_AE = compute_AE(worse, gt_xyz)
        return better_AE.cpu(), worse_AE.cpu()
    
    elif metric == 'Joint AVE':
        better, worse = jointxyz_pairs_from_filename(file_name, choise)
        gt_xyz = choose_gtxyz_dataset_from_filename(file_name)
        better = better.to(device)
        worse = worse.to(device)
        gt_xyz = gt_xyz.to(device)
        better_AVE = compute_AVE(better, gt_xyz)
        worse_AVE = compute_AVE(worse, gt_xyz)
        return better_AVE.cpu(), worse_AVE.cpu()
    
    elif metric == 'PFC':
        better, worse = jointxyz_pairs_from_filename(file_name, choise) # better shape: [3, 60, 24, 3]
        better_PFC = compute_PFC(better) # shape [3]
        worse_PFC = compute_PFC(worse)
        return better_PFC, worse_PFC
        

def calc_critic_metric(critic, metric):
    critic_diff = critic[:, 0] - critic[:, 1] # better - worse
    if metric == "Model" or metric == "MotionCritic":
        # Bigger is better!
        acc = torch.mean((critic_diff > 0).float())
    else:
        # Smaller is better!
        acc = torch.mean((critic_diff < 0).float())

    # each critic has two scores, 0 for the better and 1 for worse.
    # we want that each pair's better score and worse score go softmax to become to probablities, sum=1
    # true labels are 0
    # we want to calculate acc, log_loss and auc-roc

    target = torch.zeros(critic.shape[0], dtype=torch.long)
    # Compute log_loss
    

    # Compute probabilities with softmax

    # print(f"{critic[:20,:]}")
    probs = F.softmax(critic, dim=1).numpy()
    if metric != "Model" and metric != "MotionCritic":
        # Smaller is better!
        probs = probs[:,[1,0]]
    target_np = target.numpy()
    log_loss_value = log_loss(y_true=target_np, y_pred=probs, labels=[0, 1])
    # print(f"acc is {acc}, log_loss is {log_loss_value}")
    return acc.item(), log_loss_value
    

def critic_data_from_json(file_path, output_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    # better_motion = []
    # worse_motion = []
    total_motion = []
    category_index = {}
    cnt = 0
    for file_name, choise in tqdm(data.items()):
        if choise not in ['A', 'B', 'C', 'D']:
            continue
        if file_name.startswith("mdm"):
            cat = file_name[:4]
            idx = file_name.split('-')[2]
            label = cat + '-' + idx
            if label not in category_index.keys():
                category_index[label] = [cnt, cnt+1, cnt+2]
            else:
                category_index[label].append(cnt)
                category_index[label].append(cnt+1)
                category_index[label].append(cnt+2)
            cnt += 3
        better, worse = critic_data_from_filename(file_name, choise) # [3, 60, 25, 3]
        for i in range(better.shape[0]):
            total_motion.append({'motion_better': better[i], 'motion_worse': worse[i]})
    # print(len(category_index.keys()))
    if len(category_index) != 0:
        with open(f'data/mapping/{val_dataset}-fulleval_category.json', 'w') as file:
            json.dump(category_index, file)
    print("motion length: ", len(total_motion))
    print(f"Saving critic data to {output_path}")
    torch.save(total_motion, output_path)


def results_from_json(file_path, metric):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    better_score = []
    worse_score = []
    
    for file_name, choise in tqdm(data.items()):
        if choise not in ['A', 'B', 'C', 'D']:
            continue
        
        better_pairscore, worse_pairscore = results_from_filename(file_name, choise, metric) # [3] [3]
        better_score.append(better_pairscore)
        worse_score.append(worse_pairscore)
        
    print(f"scores' lengths are {len(better_score)}")
    better_score = torch.cat(better_score, dim=0)
    worse_score = torch.cat(worse_score, dim=0)
    both_score = torch.stack((better_score, worse_score), dim=1)
    
    
    return both_score


# Update the final execution block
if __name__ == '__main__':
    
    args = parse_args()
    
    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() else torch.device('cpu')
    
    val_dataset = args.mode
    exp_name = args.exp_name
    checkpoint = args.checkpoint
    calc_perprompt = args.calc_perprompt
    
    motion_location = os.path.join(PROJ_DIR, "datasets")
    print(f"Generated motion location is {motion_location}")
    
    # Load human annotation file and physics annotation file
    if val_dataset == 'mdmval':
        file_path = os.path.join(PROJ_DIR, f'metric/metrics_data/marked/mdm-fulleval.json')
        physics_score = np.load(os.path.join(PROJ_DIR, f'data/mpjpe/mdmval_mpjpe_norm.npy')) # (5823, 2)
    else:
        file_path = os.path.join(PROJ_DIR, f'metric/metrics_data/marked/flame-fulleval.json')
        physics_score = np.load(os.path.join(PROJ_DIR, f'data/mpjpe/flame_fulleval_mpjpe_norm.npy'))
    print(f"Evaluating, dataset is {val_dataset}, annotation file path is {file_path}")
    
    # Load or process validation motion data
    val_data_pth = os.path.join(PROJ_DIR, f'data/motion_dataset/mlist_{val_dataset}_fulleval.pth')
    if not os.path.exists(val_data_pth):
        print(f"***Processing val dataset***")
        critic_data_from_json(file_path, val_data_pth)
    
    # Load or compute PP-Motion model score
    model_score_pth = os.path.join(PROJ_DIR, f'data/scores/{exp_name}/score_{val_dataset}_{checkpoint}.npy')
    if not os.path.exists(model_score_pth):
        print(f"***Calculating PP-Motion model score***")
        model_score = get_val_scores(val_data_pth, output_pth=model_score_pth, exp_name=exp_name, ckp=checkpoint)
        np.save(model_score_pth, model_score.numpy())
    else:
        model_score = torch.from_numpy(np.load(model_score_pth))
    
        
    # Default metric: PP-Motion model
    metrics = ['Model']
    
    # Add other metrics
    if args.calc_baseline_metric:
        metrics.append('MotionCritic')
        mocritic_score_pth = os.path.join(PROJ_DIR, f'data/scores/mocritic_pretrained/score_{val_dataset}_mocritic_pre.npy')
        if not os.path.exists(mocritic_score_pth):
            print(f"***Calculating MotionCritic score***")
            mocritic_score = get_val_scores(val_data_pth, output_pth=mocritic_score_pth, exp_name="mocritic_pretrained", ckp="mocritic_pre")
        else:
            mocritic_score = torch.from_numpy(np.load(mocritic_score_pth))
    
    if args.calc_phys_metric:
        metrics.extend(['phys'])
    
    if args.calc_gt_metric:
        metrics.extend(['Joint AVE', 'Joint AE', 'Root AVE', 'Root AE', 'PFC'])
        
        # Set up ground-truth datasets
        load_gt_datasets(args.mode)
        
        # Set up ground-truth joint xyz datasets
        if not (os.path.exists("data/gt-packed/flame-gt-jointxyz.pt") and os.path.exists("data/gt-packed/uestc-gt-jointxyz.pt") and os.path.exists("data/gt-packed/humanact12-gt-jointxyz.pt")):
            print(f"building gt-xyz data")
            build_gt_xyz()
            print(f"gt-xyz data built.")
    
    
    # Prepare for calculating metrics within each prompt category
    if calc_perprompt:
        with open("data/mapping/mdm-fulleval_category.json") as f:
            mdm_category_to_idx = json.load(f)
        humanact12_keys = ["mdma-00", "mdma-01", "mdma-02", "mdma-03", "mdma-04", "mdma-05", "mdma-06", "mdma-07", "mdma-08", "mdma-09", "mdma-10", "mdma-11"]
    
        if os.path.exists('stats/metric_spearman_perprompt.csv'):
            df_spearman = pd.read_csv('stats/metric_spearman_perprompt.csv', index_col=0)
        else:
            df_spearman = pd.DataFrame()
        if os.path.exists('stats/metric_kendall_perprompt.csv'):
            df_kendall = pd.read_csv('stats/metric_kendall_perprompt.csv', index_col=0)
        else:
            df_kendall = pd.DataFrame()
        if os.path.exists('stats/metric_pearson_perprompt.csv'):
            df_pearson = pd.read_csv('stats/metric_pearson_perprompt.csv', index_col=0)
        else:
            df_pearson = pd.DataFrame()
            
    # Start calculating metrics
    for metric in metrics:
        print('### Calculating Metric:', metric)
        if metric == 'Model':
            scores = {'Model': model_score} # (batch_size, 2)
        elif metric == "MotionCritic":
            scores = {'MotionCritic': mocritic_score}
        elif metric == 'phys':
            scores_pen, scores_skate, scores_float = compute_phys(val_data_pth)
            scores = {
                'Penetration': torch.from_numpy(scores_pen), 
                'Skating': torch.from_numpy(scores_skate), 
                'Floating': torch.from_numpy(scores_float)
            }
        else:
            scores = {metric: results_from_json(file_path, metric)}
        
        # Calculate critic metrics: accuracy, log loss
        for metric_key, metric_score in scores.items():
            
            # 1. Calculate total
            print(f"evaluating total")
            acc, log_loss_value = calc_critic_metric(metric_score, metric=metric_key)
            if val_dataset == "flame":
                spearman_corr, kendall_tau, pearson_corr = metric_correlation(physics_score, metric_score.numpy(), calc_type="total", dataset=val_dataset)
            else:
                spearman_corr, kendall_tau, pearson_corr = metric_correlation(physics_score, metric_score.numpy(), calc_type="prompt", dataset=val_dataset)
            if metric_key != "Model" and metric_key != "MotionCritic":
                spearman_corr, kendall_tau, pearson_corr = -spearman_corr, -kendall_tau, -pearson_corr
            print({
                "acc": acc,
                "log_loss": log_loss_value,
                "pearson_corr": pearson_corr,
                "spearman_corr": spearman_corr,
                "kendall_corr": kendall_tau,
            })
            
            if calc_perprompt:
                # save total evaluation to dataframes
                df_spearman.loc[metric_key, 'total'] = spearman_corr
                df_kendall.loc[metric_key, 'total'] = kendall_tau
                df_pearson.loc[metric_key, 'total'] = pearson_corr
                
                # 2. Calculate humanact12 dataset perprompt
                print(f"evaluating humanact12 per prompt")
                for label in humanact12_keys:
                    idxs = mdm_category_to_idx[label]
                    physics_all = physics_score[idxs]
                    metrics_all = metric_score[idxs].numpy()
                    spearman_corr, kendall_tau, pearson_corr = metric_correlation(physics_all, metrics_all, calc_type="total")
                    if metric_key != "Model" and metric_key != "MotionCritic":
                        spearman_corr, kendall_tau, pearson_corr = -spearman_corr, -kendall_tau, -pearson_corr
                    # row: metric_key; column: prompt_label
                    df_spearman.loc[metric_key, label] = spearman_corr
                    df_kendall.loc[metric_key, label] = kendall_tau
                    df_pearson.loc[metric_key, label] = pearson_corr
            
                # 3. Calculate uestc dataset
                print(f"evaluating uestc")
                spearman_corr, kendall_tau, pearson_corr = metric_correlation(physics_score, metric_score.numpy(), calc_type="prompt", subset="uestc", dataset=val_dataset)
                if metric_key != "Model" and metric_key != "MotionCritic":
                    spearman_corr, kendall_tau, pearson_corr = -spearman_corr, -kendall_tau, -pearson_corr
                df_spearman.loc[metric_key, 'mdmu'] = spearman_corr
                df_kendall.loc[metric_key, 'mdmu'] = kendall_tau
                df_pearson.loc[metric_key, 'mdmu'] = pearson_corr
    
    if args.calc_perprompt:
        df_spearman.to_csv('stats/metric_spearman_perprompt.csv', float_format='%.3f', index=True)
        df_kendall.to_csv('stats/metric_kendall_perprompt.csv', float_format='%.3f', index=True)
        df_pearson.to_csv('stats/metric_pearson_perprompt.csv', float_format='%.3f', index=True)
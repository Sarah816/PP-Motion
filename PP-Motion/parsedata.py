import numpy as np
import lib.utils.rotation_conversions as geometry_u
import torch
import os
import sys
import json
import random

from pytorch3d.transforms import quaternion_multiply
from pytransform3d.rotations import quaternion_slerp


PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

def into_critic(generated_motion):
    # generated_raw shape is torch.Size([1, 25, 6, 60])
    root_loc = generated_motion[..., -1:, :3, :].permute(-4, -1, -3, -2)
    rot6d_motion = generated_motion[..., :-1, :, :].permute(-4, -1, -3, -2)
    axis_angle = geometry_u.matrix_to_axis_angle(geometry_u.rotation_6d_to_matrix(rot6d_motion))
    # axis_angle torch.Size([1, 60, 24, 3])
    # print(f'axis_angle {axis_angle.shape}, root_loc {root_loc.shape}')
    critic_m = torch.cat([axis_angle, root_loc], dim=-2)
    return critic_m


def interpolate_motion_slerp(motion_data, input_fps=24, target_fps=60):
    """Interpolate motion data from input_fps to target_fps using SLERP."""
    num_frames, num_joints, _ = motion_data.shape
    target_frames = int(num_frames * (target_fps / input_fps))

    # Calculate the time indices for the target frames
    time_original = torch.linspace(0, 1, num_frames)
    time_target = torch.linspace(0, 1, target_frames)

    # Initialize output tensor
    interpolated_motion = torch.zeros((target_frames, num_joints, 4))

    for joint in range(num_joints):
        joint_quaternions = motion_data[:, joint, :]
        for i in range(target_frames):
            t = time_target[i]
            low_idx = max(0, torch.searchsorted(time_original, t) - 1)
            high_idx = min(num_frames - 1, low_idx + 1)

            # print(f'frame:{i}, low_index:{low_idx}, high_index:{high_idx}')
            if low_idx == high_idx:
                interpolated_motion[i, joint, :] = joint_quaternions[low_idx]
            else:
                q0 = joint_quaternions[low_idx]
                q1 = joint_quaternions[high_idx]
                local_t = (t - time_original[low_idx]) / (time_original[high_idx] - time_original[low_idx])
                # print(f'q0:{q0}, q1:{q1}, local_t:{local_t}')
                # new_quat = slerp(q0, q1, local_t)
                new_quat = quaternion_slerp(q0, q1, local_t)
                # print(new_quat.dtype)
                if new_quat.dtype != torch.float64:
                    new_quat = torch.from_numpy(new_quat)
                interpolated_motion[i, joint, :] = new_quat
    return interpolated_motion

    
def interpolate_motion_linear(motion_data, input_fps=24, target_fps=60):
    """Interpolate motion data linearly from input_fps to target_fps.
    :Params motion_data: root location, [Frames, 1, 3]
    """
    num_frames, num_joints, _ = motion_data.shape
    target_frames = int(num_frames * (target_fps / input_fps))

    # Calculate the time indices for the target frames
    time_original = torch.linspace(0, 1, num_frames)
    time_target = torch.linspace(0, 1, target_frames)

    # Initialize output tensor
    interpolated_motion = torch.zeros((target_frames, num_joints, 3))

    for i in range(target_frames):
        t = time_target[i]
        low_idx = max(0, torch.searchsorted(time_original, t) - 1)
        high_idx = min(num_frames - 1, low_idx + 1)

        if low_idx == high_idx:
            interpolated_motion[i, 0, :] = motion_data[low_idx, 0, :]
        else:
            p0 = motion_data[low_idx, 0, :]
            p1 = motion_data[high_idx, 0, :]
            local_t = (t - time_original[low_idx]) / (time_original[high_idx] - time_original[low_idx])
            interpolated_motion[i, 0, :] = (1 - local_t) * p0 + local_t * p1

    return interpolated_motion

def change_fps(joint_quat, root_loc, original_fps=24, target_fps=60):
    '''
    :Param joint_quat: motion data in quanterion, [Frames, 24, 4]
    :Param root_loc: root location, [Frames, 1, 3]
    '''
    # print(joint_quat.shape, root_loc.shape)
    # joint_quat /= torch.norm(joint_quat, dim=-1, keepdim=True)  # Normalize quaternions
    interpolated_quat = interpolate_motion_slerp(joint_quat, original_fps, target_fps)
    interpolated_rootloc = interpolate_motion_linear(root_loc, original_fps, target_fps)
    # print(interpolated_quat.shape, interpolated_rootloc.shape)  # Output should be [150, 24, 4], [150, 1, 3]
    return interpolated_quat, interpolated_rootloc


def transform_dataset(input_motion):
    ''' 
    :Param input_motion: motion data in angle axis, [Frames, 25, 3] 
    :Returns: motion data in angle axis, [Frames, 25, 3]
    '''
    input_motion = input_motion.clone()  # 深拷贝以防止修改原始数据
    root_aa = input_motion[:, 0:1, :]
    # print('root_aa', root_aa)
    root_quat = geometry_u.axis_angle_to_quaternion(root_aa)
    rotation = torch.Tensor(
        [0.7071, 0.7071, 0, 0]
    )
    root_quat_new = quaternion_multiply(rotation, root_quat)
    joint_aa = input_motion[:, 1:-1, :]
    joint_quat =geometry_u.axis_angle_to_quaternion(joint_aa)
    joint_quat = torch.cat([root_quat_new, joint_quat], dim=1)
    
    root_loc = input_motion[:, -1:, :].clone()
    root_loc[:, :, 1], root_loc[:, :, 2] = -root_loc[:, :, 2].clone(), root_loc[:, :, 1].clone()
    joint_quat_new, root_loc_new = change_fps(joint_quat, root_loc, 24, 60)
    joint_aa_new = geometry_u.quaternion_to_axis_angle(joint_quat_new)
    motion_new = torch.cat([joint_aa_new, root_loc_new], dim = 1) 
    return motion_new

def putpair(pair_list, file_name, choise='B', type='mdm'):
    npz_file = np.load(file_name, allow_pickle=True)

    motion_list = npz_file['arr_0'].item()['motion']
    # prompt_list = npz_file['arr_0'].item()['prompt']

    processed_motion_list = []
    for motion in motion_list:
        motion = np.transpose(motion, (2,0,1))
        processed_motion_list.append(into_critic(motion))

    if type == 'mdm':
        if choise == 'A':
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[2]})
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[3]})
            return

        elif choise == 'B':
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[2]})
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[3]})
            return

        elif choise == 'C':
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[3]})
            return

        elif choise == 'D':
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[2]})
            return
        
    else:

        invalid_count = 0
        if choise == 'A':
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[0]})
            return

        elif choise == 'B':
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[1]})
            return

        elif choise == 'C':
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[2]})
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[2]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[2]})
            return

        elif choise == 'D':
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[3]})
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[3]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[3]})
            return
        

        else:
            invalid_count += 1
            if invalid_count % 20 == 0:
                print(f"invalid count reach: {invalid_count}")


    
            

def load_addfromfile(file_path, result_dict):
    with open(file_path, 'r') as file:
        data = json.load(file)
    for key, value in data.items():
        if key not in result_dict:
            result_dict[key] = value
    return result_dict


def load_addfromfolder(folder_path, result_dict):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            result_dict = load_addfromfile(file_path, result_dict)
    return result_dict

def load_shuffle(result_dict, seed=42):
    random.seed(42) 
    # Shuffle the keys in result_dict
    keys = list(result_dict.keys())
    random.shuffle(keys)

    # Create a new dictionary with shuffled keys
    shuffled_result_dict = {key: result_dict[key] for key in keys}
    return shuffled_result_dict


def put_fromdict(result_dict, pair_list, mode='full'):
    motion_dir = os.path.join(PROJ_DIR, 'data')
    
    invalid_cnt = 0
    for i, (file_name, choise) in enumerate(result_dict.items()):
        if mode == 'eval':
            if i%9 != 8:
                continue
        elif mode == 'train':
            if i%9 == 8:
                continue
        putpair(pair_list, os.path.join(motion_dir, file_name), choise, type=file_name[:3])
        if choise not in ['A', 'B', 'C', 'D']:
            invalid_cnt += 1

    print(f"this round, len is {len(result_dict.items())}, in which invalid {invalid_cnt}")
    


# for select_i in range(12):
#     result_dict = {}
#     result_dict = load_addfromfolder(f'marked/mdma/{select_i:02d}', result_dict)
#     result_dict = load_addfromfolder(f'marked/mdma-added/{select_i:02d}', result_dict)
#     pair_list = []
#     # result_dict = load_shuffle(result_dict)
#     mode = 'full' # chooing from full, train, val
#     put_fromdict(result_dict, pair_list, mode=mode)
#     pth_name = f'datasets/humanact12_{select_i:02d}-{mode}.pth'
#     torch.save(pair_list, pth_name)
#     print(f"saving .pth at {pth_name}")


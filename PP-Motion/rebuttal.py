import os
import numpy as np
import torch
from tqdm import tqdm
from parsedata import transform_dataset
from metric.correlation import metric_correlation, metric_func
from render.render import render_single
from parsedata import into_critic

def annotation_stats():
    # mpjpe = np.load("data/phys_annotation/flame_mpjpe.npy")[:300]
    # mpjpe_better = mpjpe[:, 0]
    # print(mpjpe.shape)
    # ave = np.mean(mpjpe)
    # var = np.std(mpjpe)
    # print("Average FLAME MPJPE:", ave, "Variance:", var)
    mpjpe = np.load("data/phys_annotation/mdmval_mpjpe.npy")
    print(mpjpe[[72, 73, 74, 144, 145, 146], :])
    exit(0)
    ave = np.mean(mpjpe)
    var = np.std(mpjpe)
    print("Average MDMval MPJPE:", ave, "Variance:", var)

annotation_stats()
exit(0)

def normalize_score():
    metric = "Skating"
    score = np.load(f'data/scores/prev_metrics/score_mdmval_{metric}.npy')
    norm_score = - (score - np.mean(score)) / np.std(score) # 使得某些metric score越高越好
    np.save(f'data/scores/prev_metrics/norm_score_mdmval_{metric}.npy', norm_score)
    idx_fig4a = [1853, 450, 124]
    score_fig4a = [norm_score[i] for i in idx_fig4a]
    print(f'fig4a {metric} score:', score_fig4a)
    


def vis_score():
    idx_fig4a = [1853, 450, 124]
    idx_fig4b = [95, 94, 93, 165, 166, 167]
    idx_intro = [1148, 2486]
    
    # mocritic_score_pth = 'data/scores/mocritic_pretrained/score_mdmval_mocritic_pre.npy'
    # mocritic_score = np.load(mocritic_score_pth)
    # norm_mocritic_score = (mocritic_score - np.mean(mocritic_score)) / np.std(mocritic_score)
    # np.save('data/scores/mocritic_pretrained/norm_score_mdmval_mocritic_pre.npy', norm_mocritic_score)
    # exit(0)
    
    mocritic_score_pth = 'data/scores/mocritic_pretrained/norm_score_mdmval_mocritic_pre.npy'
    mocritic_score = torch.from_numpy(np.load(mocritic_score_pth))
    
    model_score_pth = 'data/scores/norm_lossplcc_perprompt_phys0.3/norm_score_mdmval_checkpoint_latest.npy'
    model_score = torch.from_numpy(np.load(model_score_pth))
    
    fig4a_mocritic = [mocritic_score[i] for i in idx_fig4a]
    print('fig4a mocritic score:', fig4a_mocritic)
    fig4a_model = [model_score[i] for i in idx_fig4a]
    print('fig4a ours score:', fig4a_model)
    # fig4b_score = [mocritic_score[i][1] for i in idx_fig4b]
    # print('fig4b score:', fig4b_score)
    intro_score1 = mocritic_score[1148][0]
    intro_score2 = mocritic_score[2486][1]
    print('intro motioncritic score:', intro_score1, intro_score2)
    print('intro model score:', model_score[1148][0], model_score[2486][1])
    

def calc_annotation_correlation():
    physics_score = np.load("data/phys_annotation/mdmtrain_mpjpe_corrected_norm.npy") # (5832, 2) / (46740, 2)
    metric_func(torch.from_numpy(physics_score))
    human_score = np.zeros((46740, 2), dtype=float)
    human_score[:, 0] = 1.0
    s_corr, k_corr, p_corr = metric_correlation(human_score, physics_score, calc_type="prompt")
    print(f"Spearman correlation: {s_corr}, Kendall Tau: {k_corr}, Pearson correlation: {p_corr}")

def visualize_motioncritic():
    dataset_pth = "data/motion_dataset/mlist_flame.pth"
    motion_raw = torch.load(dataset_pth) # [60, 25, 3]
    motion = motion_raw[0]['motion_better']
    print(f"motion_raw shape: {motion.shape}") # [60, 25, 3]
    motion = motion.unsqueeze(dim=0) # [1, 60, 25, 3]
    motion = motion.permute(0, 2, 3, 1) # [1, 25, 3, 60]
    render_single(motion, device='cpu', comment='', file_path='render_output/flame/motion_better_0.mp4', pose_format='rotvec', no_comment=True, isYellow=True)

def visualize_mdm_gen():
    dataset_pth = "../motion-diffusion-model/data_gen/mdmgen_test_rot6d.npy"
    motion_raw = torch.from_numpy(np.load(dataset_pth)) # [batch_size, 25, 6, seqlen]
    # motion_raw = torch.load(dataset_pth) # [batch_size, 25, 6, seqlen]
    print(f"motion_raw shape: {motion_raw.shape}")
    motion = motion_raw[1].unsqueeze(dim=0) # [1, 25, 6, num_frames=60]
    motion_critic = into_critic(motion) # [1, 60, 25, 3]
    motion_critic = motion_critic.permute(0, 2, 3, 1) # [1, 25, 3, 60]
    print(motion_critic.shape)
    render_single(motion_critic, device='cpu', comment='', file_path='render_output/mdm_gen/test_1.mp4', pose_format='rotvec', no_comment=True, isYellow=True)
    
    
def process_mdmgen():
    text = "flame_text_part1"
    dataset_pth = f"../motion-diffusion-model/data_gen/mdmgen_{text}_rot6d.npy"
    motion_raw = torch.from_numpy(np.load(dataset_pth)) # [batch_size, 25, 6, seqlen]
    print(f"motion_raw shape: {motion_raw.shape}")
    motion_critic = into_critic(motion_raw) # [batch_size, 60, 25, 3]
    output_pth = f"data/mdm_gen/mdmgen_{text}_mocritic.npy"
    print(f'Saving data to {output_pth}, shape {motion_critic.shape}')
    np.save(output_pth, motion_critic.numpy()) # Save as numpy array
    
    data_len = len(motion_critic)
    data_processed = []
    data_processed = np.zeros((data_len, 150, 25, 3))
    for i in tqdm(range(data_len)):
        motion = motion_critic[i]
        assert(motion.shape == (60, 25, 3))
        data_processed[i] = transform_dataset(motion)
        
    data_processed = torch.from_numpy(data_processed) # [batch_size, 150, 25, 3]
    torch.save(data_processed, f"data/mdm_gen/mdmgen_{text}_processed.pth")
    

if __name__ == "__main__":
    # annotation_stats()
    # normalize_score()
    # vis_score()
    # calc_annotation_correlation()
    # visualize_mdm_gen()
    # visualize_motioncritic()
    process_mdmgen()
    
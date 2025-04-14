import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm
import json
import sys
import os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

from lib.model.critic import MotionCritic

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.backends import cudnn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import average_precision_score, brier_score_loss, accuracy_score


def metric_func(critic):
    # critic's shape is [batch_size,2]
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = F.cross_entropy(critic, target, reduction='none')
    loss = torch.mean(loss_list)
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    # each critic has two scores, 0 for the better and 1 for worse.
    # we want that each pair's better score and worse score go softmax to become to probablities, sum=1
    # true labels are 0
    # we want to calculate acc, log_loss and auc-roc

    # Compute probabilities with softmax
    probs = F.softmax(critic, dim=1).cpu().detach().numpy()
    target_np = target.cpu().numpy()
    # Compute log_loss
    log_loss_value = log_loss(y_true=target_np, y_pred=probs, labels=[0, 1])
    print(f"acc is {acc}, log_loss is {log_loss_value}")


# critic_score = np.load("stats/norm_lossplcc_perprompt_phys0.3_mdmval.npy") # (5823, 2) 用MotionCritic所给的原始模型进行评测
# physics_score= np.load("data/mpjpe/mdmval_mpjpe.npy")
# critic_worse = critic_score[:, 1]
# critic_better = critic_score[:, 0]
# with open("data/mapping/mdmval_category.json") as f:
#     category_to_idx = json.load(f)
# phys = []
# for cate, idxs in category_to_idx.items():
#     if cate.startswith("mdmu"):
#         continue
#     idxs = np.array(idxs, dtype=int)
#     critic_all = np.concatenate((critic_better[idxs],
#                                 critic_worse[idxs]))
    
# print(np.mean(phys))
# exit(0)

def metric_correlation(critic_score, physics_score, calc_type, subset=None):
    critic_worse = critic_score[:, 1]
    critic_better = critic_score[:, 0]
    physics_worse = physics_score[:, 1]
    physics_better = physics_score[:, 0]
    val_len = len(critic_score)
    s_corr, k_corr, p_corr = [],[],[]
    if calc_type == "pair":
        print("---Calculate unit num: 2---")
        for i in tqdm(range(val_len)):
            critic_all = np.array([critic_better[i], critic_worse[i]])
            physics_all = np.array([physics_better[i], physics_worse[i]])
            spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
            kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
            pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
            # print(spearman_corr, kendall_tau, pearson_corr)
            s_corr.append(spearman_corr)
            k_corr.append(kendall_tau)
            p_corr.append(pearson_corr)
        return sum(s_corr)/len(s_corr), sum(k_corr)/len(k_corr), sum(p_corr)/len(p_corr)
    
    elif calc_type == "quat":
        print("---Calculate unit num: 4---")
        for i in tqdm(range(int(val_len / 3))):
            critic_all = np.array([critic_better[i*3], critic_worse[i*3], critic_worse[i*3+1], critic_worse[i*3+2]])
            physics_all = np.array([physics_better[i*3], physics_worse[i*3], physics_worse[i*3+1], physics_worse[i*3+2]])
            # print(critic_all, physics_all)
            spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
            kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
            pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
            # print(spearman_corr, kendall_tau, pearson_corr)
            s_corr.append(spearman_corr)
            k_corr.append(kendall_tau)
            p_corr.append(pearson_corr)
        return sum(s_corr)/len(s_corr), sum(k_corr)/len(k_corr), sum(p_corr)/len(p_corr)
    
    elif calc_type == "batch":
        print("---Calculate unit: batch---")
        batch_size = 100  # 每次处理 32 条 better + 32 条 worse 数据
        num_batches = val_len // batch_size  # 计算批次数 182
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, val_len)
            # 取出 32 条 better 和 32 条 worse 数据
            critic_all = np.concatenate((critic_better[start_idx:end_idx],
                                        critic_worse[start_idx:end_idx]))
            physics_all = np.concatenate((physics_better[start_idx:end_idx],
                                        physics_worse[start_idx:end_idx]))
            # 计算相关性
            spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
            kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
            pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
            s_corr.append(spearman_corr)
            k_corr.append(kendall_tau)
            p_corr.append(pearson_corr)
        return sum(s_corr)/len(s_corr), sum(k_corr)/len(k_corr), sum(p_corr)/len(p_corr)
    
    elif calc_type == "prompt":
        # print("---Calculate unit: prompt---")
        with open("data/mapping/mdm-fulleval_category.json") as f: # TODO: refactor
            category_to_idx = json.load(f)
        for cate, idxs in category_to_idx.items():
            if subset == "uestc":
                if cate.startswith("mdma"):
                    continue
            idxs = np.array(idxs, dtype=int)
            critic_all = critic_score[idxs].reshape(-1)
            physics_all = physics_score[idxs].reshape(-1)
            assert(critic_all.shape == physics_all.shape)
            spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
            kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
            pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
            s_corr.append(spearman_corr)
            k_corr.append(kendall_tau)
            p_corr.append(pearson_corr)
        return sum(s_corr)/len(s_corr), sum(k_corr)/len(k_corr), sum(p_corr)/len(p_corr)
    
    elif calc_type == "total":
        # critic_all = np.concatenate((critic_better[::3], critic_worse)) # 只取不重复数据
        # physics_all = np.concatenate((physics_better[::3], physics_worse))
        critic_all = critic_score.reshape(-1)
        physics_all = physics_score.reshape(-1)
        assert(critic_all.shape == physics_all.shape)
        spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
        kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
        pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
        return spearman_corr, kendall_tau, pearson_corr


if __name__ == "__main__":
    
    # critic_score = np.load("stats/norm_lossplcc.npy") # (5823, 2) 用MotionCritic所给的原始模型进行评测
    critic_score = np.load("stats/norm_lossplcc_perprompt_phys0.3_mdmval.npy") # (5823, 2) 用MotionCritic所给的原始模型进行评测
    physics_score= np.load("data/mpjpe/mdmval_mpjpe.npy")
    
    # calc_type = "pair", "quat", "batch", "prompt", "total"
    spearman_corr, kendall_tau, pearson_corr, spearman_p, kendall_p, pearson_p = metric_correlation(critic_score, physics_score, calc_type="total")
    print("spearman corr: ", spearman_corr, " p: ", spearman_p)
    print("kendall tau: ", kendall_tau, " p: ", kendall_p)
    print("pearson corr: ", pearson_corr, " p: ", pearson_p)
    
    
    
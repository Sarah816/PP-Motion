import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from tqdm import tqdm
import json

# def min_max_normalize_and_reverse(data):
#     """
#     Min-Max 归一化并颠倒大小顺序
#     :param data: 输入的 numpy 数组
#     :return: 归一化并颠倒顺序后的数组
#     """
#     # Min-Max 归一化
#     min_val = np.min(data)
#     max_val = np.max(data)
#     print(min_val, max_val)
#     normalized_data = (data - min_val) / (max_val - min_val)
    
#     # 颠倒大小顺序
#     reversed_data = 1 - normalized_data
    
#     return reversed_data




if __name__ == "__main__":
    # val_physics_worse = np.load("data/mpjpe/mdmval_mpjpe_worse.npy") # 5823
    # print(val_physics_worse[:30])
    # val_physics_better = np.load("data/mpjpe/mdmval_mpjpe_better.npy") # 5823
    # train_physics_worse = np.load("data/mpjpe/mdmtrain_mpjpe_worse.npy") # 43670
    # train_physics_better = np.load("data/mpjpe/mdmtrain_mpjpe_better.npy") # 43670
    # mpjpe = np.concatenate((val_physics_worse, val_physics_better, train_physics_worse, train_physics_better))
    # phys_score = min_max_normalize_and_reverse(mpjpe)
    # print(phys_score[:30])
    # exit(0)
    # np.save("data/mpjpe/mdmval_physcore_worse.npy", phys_score[:5823])
    # np.save("data/mpjpe/mdmval_physcore_better.npy", phys_score[5823:11646])
    # np.save("data/mpjpe/mdmtrain_physcore_worse.npy", phys_score[11646:55316])
    # np.save("data/mpjpe/mdmtrain_physcore_better.npy", phys_score[55316:98986])
    # exit(0)
    
    
    
    critic_score = np.load("stats/critic_score_val.npy") # (5823, 2) 用MotionCritic所给的原始模型进行评测
    critic_worse = critic_score[:, 1]
    # for i in range(int(critic_better.shape[0] / 3)):
    #     assert(critic_better[i*3+1]==critic_better[i*3+2])
    #     assert(critic_better[i*3+1]==critic_better[i*3])
    critic_better = critic_score[:, 0] # 相邻三条better数据相同
    physics_score= np.load("data/mpjpe/mdmval_mpjpe.npy")
    physics_worse = physics_score[:, 1]
    physics_better = physics_score[:, 0]
    val_len = len(critic_score)
    # print(critic_all.shape, physics_all.shape) # (7764,)
    # p_value 相关性的统计显著性，p < 0.05 表示相关性显著。样本数量越多，相关性越强，p通常越小。
    
    # 每两条数据（一条better一条worse）计算相关性，然后取均值
    # s_corr, k_corr, p_corr, s_p, k_p, p_p = [],[],[],[],[],[]
    # for i in tqdm(range(val_len)):
    #     critic_all = np.array([critic_better[i], critic_worse[i]])
    #     physics_all = np.array([physics_better[i], physics_worse[i]])
    #     spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
    #     kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
    #     pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
    #     # print(spearman_corr, kendall_tau, pearson_corr)
    #     s_corr.append(spearman_corr)
    #     k_corr.append(kendall_tau)
    #     p_corr.append(pearson_corr)
    #     s_p.append(spearman_p)
    #     k_p.append(kendall_p)
    #     p_p.append(pearson_p)
    # # print(sum(s_corr[300:330])/30)
    # print("spearman corr: ", sum(s_corr)/len(s_corr), " p: ", sum(s_p)/len(s_p))
    # print("kendall tau: ", sum(k_corr)/len(k_corr), " p: ", sum(k_p)/len(k_p))
    # print("pearson corr: ", sum(p_corr)/len(p_corr), " p: ", sum(p_p)/len(p_p))
    
    
    # 每四条数据（三条better一条worse）计算相关性，然后取均值
    # s_corr, k_corr, p_corr, s_p, k_p, p_p = [],[],[],[],[],[]
    # for i in tqdm(range(int(val_len / 3))):
    #     critic_all = np.array([critic_better[i*3], critic_worse[i*3], critic_worse[i*3+1], critic_worse[i*3+2]])
    #     physics_all = np.array([physics_better[i*3], physics_worse[i*3], physics_worse[i*3+1], physics_worse[i*3+2]])
    #     # print(critic_all, physics_all)
    #     spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
    #     kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
    #     pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
    #     # print(spearman_corr, kendall_tau, pearson_corr)
    #     s_corr.append(spearman_corr)
    #     k_corr.append(kendall_tau)
    #     p_corr.append(pearson_corr)
    #     s_p.append(spearman_p)
    #     k_p.append(kendall_p)
    #     p_p.append(pearson_p)
    # # print(sum(s_corr[300:330])/30)
    # print("spearman corr: ", np.mean(s_corr), " p: ", np.mean(s_p))
    # print("kendall tau: ", np.mean(k_corr), " p: ", np.mean(k_p))
    # print("pearson corr: ", np.mean(p_corr), " p: ", np.mean(p_p))
    # exit(0)
    
    # 所有prompt相同的数据计算相关性，然后取均值
    s_corr, k_corr, p_corr, s_p, k_p, p_p = [],[],[],[],[],[]
    batch_size = 32  # 每次处理 32 条 better + 32 条 worse 数据
    num_batches = val_len // batch_size  # 计算批次数 182
    with open("data/mapping/mdmval_category.json") as f:
        category_to_idx = json.load(f)
    for cate, idxs in category_to_idx.items():
        idxs = np.array(idxs, dtype=int)
        critic_all = np.concatenate((critic_better[idxs],
                                    critic_worse[idxs]))
        physics_all = np.concatenate((physics_better[idxs],
                                    physics_worse[idxs]))
        print(critic_all.shape, physics_all.shape)
        
        spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
        kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
        pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
        s_corr.append(spearman_corr)
        k_corr.append(kendall_tau)
        p_corr.append(pearson_corr)
        s_p.append(spearman_p)
        k_p.append(kendall_p)
        p_p.append(pearson_p)
    print("spearman corr: ", np.mean(s_corr), " p: ", np.mean(s_p))
    print("kendall tau: ", np.mean(k_corr), " p: ", np.mean(k_p))
    print("pearson corr: ", np.mean(p_corr), " p: ", np.mean(p_p))
    exit(0)
    
    # 每64条数据（一个batch，包含32条better和32条worse）计算相关性，然后取均值
    s_corr, k_corr, p_corr, s_p, k_p, p_p = [],[],[],[],[],[]
    batch_size = 32  # 每次处理 32 条 better + 32 条 worse 数据
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
        s_p.append(spearman_p)
        k_p.append(kendall_p)
        p_p.append(pearson_p)
    print("spearman corr: ", np.mean(s_corr), " p: ", np.mean(s_p))
    print("kendall tau: ", np.mean(k_corr), " p: ", np.mean(k_p))
    print("pearson corr: ", np.mean(p_corr), " p: ", np.mean(p_p))
    exit(0)
    
    # 计算整体相关性
    critic_all = np.concatenate((critic_better[::3], critic_worse)) # 只取不重复数据
    physics_all = np.concatenate((physics_better[::3], physics_worse))
    spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
    kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
    pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
    print("spearman corr: ", spearman_corr, " p: ", spearman_p) # -0.3318407717327049 6.224797408377744e-199
    print("kendall tau: ", kendall_tau, " p: ", kendall_p)
    print("pearson corr: ", pearson_corr, " p: ", pearson_p)
    
    
import json
import numpy as np
import torch
from tqdm import tqdm


np.set_printoptions(precision=3, floatmode='fixed', suppress=True)

def get_finetune_subset():
    with open("data/mdmtrain_compare.json") as f:
        match_dataset = json.load(f)
    total = set(range(46740))
    exist = []
    for idx_new, idx_old in match_dataset:
        exist.append(idx_new)
    finetune_subset = total - set(exist)
    print(len(finetune_subset)) # 3070
    with open('data/finetune_index.json', 'w') as f:
        json.dump(sorted(finetune_subset), f)
        
        

def reconstruct_dataset_overlap():
    with open("data/mdmtrain_compare.json") as f:
        match_dataset = json.load(f)
    new_len = len(match_dataset)
    print("match length: ", new_len)
    mpjpe_better = np.load("data/mpjpe/mdmtrain_old_mpjpe_better.npy")
    mpjpe_worse = np.load("data/mpjpe/mdmtrain_old_mpjpe_worse.npy")
    mpjpe_better_new = np.zeros(new_len, dtype=np.float32)
    mpjpe_worse_new = np.zeros(new_len, dtype=np.float32)
    dataset_new = torch.load("data/motion_dataset/mlist_mdmfull_train_corrected.pth")
    dataset_overlap = []
    for i in tqdm(range(new_len)):
        idx_new, idx_old = match_dataset[i]
        dataset_overlap.append(dataset_new[idx_new])
        mpjpe_better_new[i] = mpjpe_better[idx_old]
        mpjpe_worse_new[i] = mpjpe_worse[idx_old]
    # torch.save(dataset_overlap, "data/motion_dataset/mlist_mdmfull_train_overlap.pth")
    np.save("data/mpjpe/mdmtrain_mpjpe_better.npy", mpjpe_better_new)
    np.save("data/mpjpe/mdmtrain_mpjpe_worse.npy", mpjpe_worse_new)
        
def check_better():
    data = torch.load("data/motion_dataset/mlist_mdmfull_train_corrected.pth")
    for i in tqdm(range(0, len(data), 3)):
        better1 = data[i]["motion_better"]
        better2 = data[i+1]["motion_better"]
        better3 = data[i+2]["motion_better"]
        if not (torch.all(better1 == better2) and torch.all(better1 == better3)):
            print(f"Error index: {i}")
        
def reconstruct_dataset():
    total_len = 46740
    total_mpjpe = np.zeros((total_len, 2))
    current_better = np.load("data/mpjpe/mdmtrain_mpjpe_better.npy")
    current_worse = np.load("data/mpjpe/mdmtrain_mpjpe_worse.npy")
    flag = 0
    with open('data/finetune_index.json', 'r') as f:
        finetune_index = json.load(f)
    new_better = np.load("data/mpjpe/mpjpe_better_new.npy")
    new_worse = np.load("data/mpjpe/mpjpe_worse_new.npy")
    for idx in tqdm(finetune_index):
        total_mpjpe[idx][0] = new_better[flag]
        total_mpjpe[idx][1] = new_worse[flag]
        flag+=1
    overlap_index = set(range(total_len)) - set(finetune_index)
    overlap_index = sorted(list(overlap_index))
    assert(len(overlap_index) == current_better.shape[0])
    flag = 0
    for idx in tqdm(overlap_index):
        total_mpjpe[idx][0] = current_better[flag]
        total_mpjpe[idx][1] = current_worse[flag]
    missing_indices = np.where(total_mpjpe[:] == 0)[0]
    assert(len(missing_indices) == 0)
    total_mpjpe = total_mpjpe.astype(np.float32)
    np.save("data/mpjpe/mdmtrain_mpjpe.npy", total_mpjpe)
        
def reconstruct_mpjpe():
    total_len = 46740
    total_mpjpe = np.zeros((total_len, 2))
    old_mpjpe = np.load("data/mpjpe/mdmtrain_mpjpe_old.npy")
    new_mpjpe = np.load("data/mpjpe/mdmtrain_mpjpe_new.npy")
    print(old_mpjpe.shape, new_mpjpe.shape)
    with open('data/finetune_index.json', 'r') as f1:
        finetune_index = json.load(f1)
    with open('data/mdmtrain_compare.json', 'r') as f2:
        overlap_pair = json.load(f2)
    assert len(finetune_index) == new_mpjpe.shape[0]
    for flag in tqdm(range(len(finetune_index))):
        idx = finetune_index[flag]
        total_mpjpe[idx] = new_mpjpe[flag]
    for flag in tqdm(range(len(overlap_pair))):
        new_idx = overlap_pair[flag][0]
        old_idx = overlap_pair[flag][1]
        assert np.array_equal(total_mpjpe[new_idx], np.array([0, 0]))
        total_mpjpe[new_idx] = old_mpjpe[old_idx]
    missing_indices = np.where(total_mpjpe[:] == 0)[0]
    assert(len(missing_indices) == 0)
    np.save("data/mpjpe/mdmtrain_mpjpe_corrected.npy", total_mpjpe.astype(np.float32))

# better = np.load("data/mpjpe/mpjpe_better_new.npy")
# worse = np.load("data/mpjpe/mpjpe_worse_new.npy")
# total_mpjpe = np.stack((better, worse), axis=-1).astype(np.float32)
# print(total_mpjpe.shape)
# np.save("data/mpjpe/mdmtrain_mpjpe_new.npy", total_mpjpe)
# exit(0)

def plot_distribution():
    total_mpjpe = np.load("data/mpjpe/mdmtrain_mpjpe.npy")
    # print(np.max(total_mpjpe[:, 0]))
    # print(np.min(total_mpjpe[:, 0]))
    # print(np.mean(total_mpjpe[:, 0]))
    # print(np.max(total_mpjpe[:, 1]))
    # print(np.min(total_mpjpe[:, 1]))
    # print(np.mean(total_mpjpe[:, 1]))
    import seaborn as sns
    import matplotlib.pyplot as plt
    # 绘制核密度估计图
    # sns.kdeplot(total_mpjpe, color='red', fill=True)
    # plt.title("Kernel Density Estimate of Data Distribution")
    # plt.xlabel("Value")
    # plt.ylabel("Density")
    # plt.grid(True)
    # plt.savefig("pics/mpjpe_kernel.png", dpi=300)

    sns.boxplot(x=total_mpjpe[:, 1], color='green')
    plt.title("Boxplot of Data Distribution")
    plt.xlabel("Value")
    plt.grid(True)
    plt.savefig("pics/mpjpe_boxplot_worse.png", dpi=300, bbox_inches='tight')
    


def hash_array(array, decimals=6):
    """
    生成 NumPy 数组的哈希值，处理浮点数精度问题。

    参数:
        array (np.array): 输入的 NumPy 数组。
        decimals (int): 四舍五入的小数位数。

    返回:
        int: 哈希值。
    """
    # 对浮点数四舍五入
    array_rounded = np.round(array, decimals=decimals)
    # 将数组展平并转换为元组
    array_tuple = tuple(array_rounded.flatten())
    # 生成哈希值
    return hash(array_tuple)

def check_duplication():
    dataset1 = torch.load("data/motion_dataset/mlist_flame.pth")
    dataset2 = torch.load("data/motion_dataset/mlist_flame_fulleval.pth")
    data_len = len(dataset1)
    for i in tqdm(range(data_len)):
        tensor1 = dataset1[i]["motion_better"]
        tensor2 = dataset2[i]["motion_better"]
        same = np.isclose(tensor1, tensor2, rtol=1e-5, atol=1e-8).all()
        assert(tensor1.shape == tensor2.shape)
        assert(same)
        

def map_flame_eval():
    # dataset1 = torch.load("data/motion_dataset/mlist_flame.pth")
    # dataset2 = torch.load("data/motion_dataset/mlist_flame_fulleval.pth")
    # data_len = len(dataset1)
    # mapping = []
    # for i in tqdm(range(data_len)):
    #     tensor1 = dataset1[i]["motion_better"]
    #     for j in range(data_len):
    #         tensor2 = dataset2[j]["motion_better"]
    #         same = np.isclose(tensor1, tensor2, rtol=1e-5, atol=1e-8).all()
    #         if same:
    #             mapping.append([i, j])
    #             worse_same = np.isclose(dataset1[i]["motion_worse"], dataset2[j]["motion_worse"], rtol=1e-5, atol=1e-8).all()
    #             assert(worse_same)
    #             break
    # print(len(mapping)) # 603
    # with open("data/mapping/flame_compare.json", 'w') as f:
    #     json.dump(mapping, f)
    
    
    with open("data/mapping/flame_compare.json") as f:
        mapping = json.load(f)
    mpjpe = np.load("data/mpjpe/flame_mpjpe.npy")
    new_mpjpe = np.zeros((603, 2))
    for index_old, index_new in mapping:
        new_mpjpe[index_new] = mpjpe[index_old]
    np.save("data/mpjpe/flame_fulleval_mpjpe.npy", new_mpjpe)
        

def find_duplicate_indices():
    dataset = torch.load("data/motion_dataset/mlist_mdmtrain_corrected.pth")
    array_to_indices = {}
    # print(torch.all(dataset[0]["motion_better"] == dataset[1]["motion_better"]))
    # tensor1 = dataset[0]["motion_better"]
    # tensor2 = dataset[1]["motion_better"]
    # hash1 = hash(tensor1.numpy().tobytes())
    # hash2 = hash(tensor2.numpy().tobytes())
    # print(hash1 == hash2)
    # exit(0)
    # for i in tqdm(range(10)):
    for i in tqdm(range(len(dataset))):
        motion_better = dataset[i]["motion_better"]
        fingerprint = hash(motion_better.numpy().tobytes())
        if fingerprint in array_to_indices:
            # print("find duplicate")
            array_to_indices[fingerprint].append(i)
            if len(array_to_indices[fingerprint]) > 3:
                print(f"Error in indexes: {array_to_indices[fingerprint]}")
        else:
            array_to_indices[fingerprint] = [i]
        # motion_tuple = tuple(motion_better.flatten())
        # print(motion_better)
        # print(motion_tuple)
        # found = False
        # for existing_tuple in array_to_indices:
        #     if np.isclose(motion_better, np.array(existing_tuple).reshape(motion_better.shape), rtol=1e-5, atol=1e-8).all():
        #         # print("find duplicate")
        #         array_to_indices[existing_tuple].append(i)
        #         found = True
        #         break
        
        # if not found:
        #     array_to_indices[motion_tuple] = [i]
        # print(motion_better.shape)
    duplicates = array_to_indices.values()
    print(len(duplicates)) # 16745
    with open("data/motion_better_duplicate.json", 'w') as f:
        json.dump(list(duplicates), f, indent=4)
    
    return duplicates
    
def categorize_by_label():
    with open("data/mapping/mdmtrain_prompt.json") as f:
        labels = json.load(f)
    category_to_idx = {}
    for key, value in tqdm(labels.items()):
        if value in category_to_idx:
            category_to_idx[value].append(key)
        else:
            category_to_idx[value] = [key]
    for key, value in category_to_idx.items():
        print(key, len(value))
    # with open("data/mapping/mdmtrain_category.json", 'w') as f1:
    #     json.dump(category_to_idx, f1, indent=4)
    print('total category nums', len(category_to_idx.items()))
    
def normalize_mpjpe():
    data = np.load("data/mpjpe/mdmval_mpjpe.npy")
    mean = np.mean(data)
    std = np.std(data)
    data_norm = - (data - mean) / std # 标准化为mean=0，std=1的数据分布，并且颠倒顺序
    print(np.mean(data_norm), np.std(data_norm))
    np.save("data/mpjpe/mdmval_mpjpe_norm.npy", data_norm)

def normalize_score():
    mdmval_score = np.load("data/scores/norm_lossplcc_perprompt_phys0.3/score_mdmval_checkpoint_latest.npy")
    # flame_score = np.load("data/scores/norm_lossplcc_perprompt_phys0.3/score_flame_checkpoint_latest.npy")
    # total_score = np.concatenate([mdmval_score, flame_score])
    mdmval_mean = np.mean(mdmval_score)
    mdmval_std = np.std(mdmval_score)
    before_ft = (-4.35 - mdmval_mean) / mdmval_std
    after_ft = (4.09 - mdmval_mean) / mdmval_std
    print(before_ft, after_ft)
    # flame_mean = np.mean(flame_score)
    # flame_std = np.std(flame_score)
    # mean = np.mean(total_score)
    # std = np.std(total_score)
    print(mdmval_mean, mdmval_std)
    data_norm = (mdmval_score - mdmval_mean) / mdmval_std
    # np.save("data/scores/norm_lossplcc_perprompt_phys0.3/norm_score_mdmval_checkpoint_latest.npy", data_norm)
    indexes = [124,450,1853,93,94,95,165,166,167]
    print(data_norm[indexes])

def dataset_stat():
    # 统计数据集的均值和标准差
    data = np.load("data/mpjpe/mdmtrain_mpjpe_corrected.npy")
    mean = np.mean(data)
    std = np.std(data)
    print("Mean: ", mean)
    print("Std: ", std)
    print("Max: ", np.max(data))
    print("Min: ", np.min(data))
    print("Shape: ", data.shape)

def stat_for_visualize():
    score = np.load("data/scores/norm_lossplcc_perprompt_phys0.3/score_mdmval_checkpoint_latest.npy")
    mpjpe = np.load("data/mpjpe/mdmval_mpjpe.npy")
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)  # 固定保留2位小数
    idx = []
    for i in range(500, 1000):
        worse1 = score[3*i][1]
        worse2 = score[3*i+1][1]
        worse3 = score[3*i+2][1]
        mpjpe1 = mpjpe[3*i][1]
        mpjpe2 = mpjpe[3*i+1][1]
        mpjpe3 = mpjpe[3*i+2][1]
        std1 = np.var([worse1, worse2, worse3])
        std2 = np.var([mpjpe1, mpjpe2, mpjpe3])
        if std1 >= 100 and std2 >= 100:
            idx.append(3*i)
            idx.append(3*i+1)
            idx.append(3*i+2)
            print(f"worse index: {3*i}")
            print(f"score: {worse1:.2f}, {worse2:.2f}, {worse3:.2f}")
            print(f"mpjpe: {mpjpe1:.2f}, {mpjpe2:.2f}, {mpjpe3:.2f}")
        
        # if score[i][1] - score[i][0] > 8 and score[i][1] > 3:
        #     print(i, ': ', score[i])
        #     idx.append(i)
        
        # if 26 < mpjpe[i][1] < 28:
        #     print(i, ': ', mpjpe[i][0], mpjpe[i][1])
        #     idx.append(i)
    print(idx)
    
if __name__ == "__main__":
    # reconstruct_dataset()
    # get_finetune_subset()
    # check_better()
    # reconstruct_dataset()
    # find_duplicate_indices()
    # reconstruct_mpjpe()
    # categorize_by_label()
    # normalize_mpjpe()
    # check_duplication()
    # dataset_stat()
    stat_for_visualize()
    # map_flame_eval()
    
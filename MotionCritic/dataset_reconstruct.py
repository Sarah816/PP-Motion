import json
import numpy as np
import torch
from tqdm import tqdm

# dataset_new = torch.load("data/mlist_mdmfull_train_overlap.pth")
# print(len(dataset_new))
# exit(0)

# mpjpe_better = np.load("data/mpjpe/mdmval_mpjpe_better_short.npy")
# len_old = len(mpjpe_better)
# mpjpe_better_new = np.zeros(len_old*3, dtype=np.float32)
# for i in range(mpjpe_better.shape[0]):
#     mpjpe_better_new[i*3] = mpjpe_better[i]
#     mpjpe_better_new[i*3+1] = mpjpe_better[i]
#     mpjpe_better_new[i*3+2] = mpjpe_better[i]
# print(mpjpe_better[:5])
# print(mpjpe_better_new[:15])
# np.save("data/mpjpe/mdmval_mpjpe_better.npy", mpjpe_better_new)
# exit(0)

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
    dataset_new = torch.load("data/mlist_mdmfull_train_corrected.pth")
    dataset_overlap = []
    for i in tqdm(range(new_len)):
        idx_new, idx_old = match_dataset[i]
        dataset_overlap.append(dataset_new[idx_new])
        mpjpe_better_new[i] = mpjpe_better[idx_old]
        mpjpe_worse_new[i] = mpjpe_worse[idx_old]
    # torch.save(dataset_overlap, "data/mlist_mdmfull_train_overlap.pth")
    np.save("data/mpjpe/mdmtrain_mpjpe_better.npy", mpjpe_better_new)
    np.save("data/mpjpe/mdmtrain_mpjpe_worse.npy", mpjpe_worse_new)
        
def check_better():
    data = torch.load("data/mlist_mdmfull_train_corrected.pth")
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


def find_duplicate_indices():
    dataset = torch.load("data/mlist_mdmtrain_corrected.pth")
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
    
    
if __name__ == "__main__":
    # reconstruct_dataset()
    # get_finetune_subset()
    # check_better()
    # reconstruct_dataset()
    # find_duplicate_indices()
    # reconstruct_mpjpe()
    # categorize_by_label()
    normalize_mpjpe()
    
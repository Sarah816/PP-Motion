import sys
import os
PROJ_DIR = os.path.dirname(os.path.abspath(__file__)) # 'MotionCritic/MotionCritic'
sys.path.append(PROJ_DIR)
os.environ['WANDB__EXECUTABLE'] =  '/home/zhaosh/miniconda3/envs/mocritic/bin/python'
os.environ['WANDB_DIR'] = PROJ_DIR + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = PROJ_DIR + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = PROJ_DIR + '/wandb/.config/'
import numpy as np
import time

from lib.model.critic import MotionCritic

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.backends import cudnn
import gc
import pytorch_warmup as warmup
import argparse
import wandb
import torchsort
import json
from scipy.stats import spearmanr, pearsonr, kendalltau

# this might be useful
torch.manual_seed(3407)

checkpoint_interval = 15

# mpjpe = np.load(os.path.join(PROJ_DIR, 'data/mpjpe/mdmval_mpjpe.npy'))
# mpjpe = - mpjpe
# np.save(os.path.join(PROJ_DIR, 'data/mpjpe/mdmval_mpjpe_reverse.npy'), mpjpe)
# exit(0)

def parse_args():

    # Initialize argparse
    parser = argparse.ArgumentParser(description='Your description here')

    # Add command-line arguments
    parser.add_argument('--gpu_indices', type=str, default="0,1",  # Change this to the GPU indices you want to use
                        help='Indices of the GPUs to use, separated by commas (default: "0,1")')


    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for training (default: 32)')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 32)')

    parser.add_argument('--epoch', type=int, default=120,
                        help='Batch size for training (default: 32)')

    parser.add_argument('--exp_name', type=str, default="exp7_2e-5_decay_seqsplit",
                        help='Experiment name for WandB')

    parser.add_argument('--dataset', type=str, default="hfull_shuffle",
                        help='to determine use which train and val dataset')

    # arguments
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='Whether to save model checkpoints during training (default: False)')

    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to the saved model checkpoint to load (default: None)')

    parser.add_argument('--save_latest', action='store_true',
                        help='Whether to save model checkpoints during training (default: False)')

    parser.add_argument('--lr_warmup', action='store_true',
                        help='Whether to save model checkpoints during training (default: False)')

    parser.add_argument('--lr_decay', action='store_true',
                        help='Whether to save model checkpoints during training (default: False)')

    parser.add_argument('--big_model', action='store_true',
                        help='Initiate a bigger model')

    parser.add_argument('--origin_model', action='store_true',
                        help='use not sigmoid')

    parser.add_argument('--debug', action='store_true',
                        help='debug mode, no wandb')
    
    parser.add_argument('--enable_phys', action='store_true',
                        help='enabel physics model')
    
    parser.add_argument('--phys_bypass', action='store_true',
                        help='enabel physics bypass')
    parser.add_argument('--loss_type', type=str, default="mse",
                        help='mse/plcc/srocc')
    parser.add_argument('--critic_coef', type=float, default=1.0,
                        help='critic loss coef')
    parser.add_argument('--phys_coef', type=float, default=1.0,
                        help='phys loss coef')


    # Parse the arguments
    return parser.parse_args()


def create_data_loaders(dataset, batch_size):
    train_motion_pairs = motion_pair_dataset(dataset_name="mdmtrain", dataset_type=dataset)
    train_loader = DataLoader(train_motion_pairs, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    # val_motion_pairs = motion_pair_dataset(dataset_name="mdmval", dataset_type=dataset)
    # val_loader = DataLoader(val_motion_pairs, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_motion_pairs = MotionCategoryDataset(dataset_name="mdmval", dataset_type=dataset)
    val_loader = DataLoader(val_motion_pairs, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    return train_loader, val_loader

class MotionCategoryDataset(Dataset):
    def __init__(self, dataset_name, dataset_type):
        if dataset_name == "mdmtrain":
            motion_dataset_pth = os.path.join(PROJ_DIR, f'data/mlist_{dataset_name}_{dataset_type}.pth')
        elif dataset_name == "mdmval":
            motion_dataset_pth = os.path.join(PROJ_DIR, f'data/mlist_{dataset_name}.pth')
        else:
            raise ValueError("Unsupported dataset name.")
            
        print(f"Loading dataset from {motion_dataset_pth}")
        self.data = torch.load(motion_dataset_pth)

        if enable_phys:
            if dataset_name == "mdmtrain":
                mpjpe_path = os.path.join(PROJ_DIR, f'data/mpjpe/{dataset_name}_mpjpe_{dataset_type}_norm.npy')
            elif dataset_name == "mdmval":
                mpjpe_path = os.path.join(PROJ_DIR, f'data/mpjpe/{dataset_name}_mpjpe_norm.npy')
            print(f"Loading mpjpe from {mpjpe_path}")
            mpjpe = np.load(mpjpe_path)
            for i in range(len(self.data)):
                self.data[i]['mpjpe_better'] = mpjpe[i][0]
                self.data[i]['mpjpe_worse'] = mpjpe[i][1]
        print(f"Dataset {dataset_name} length: {len(self.data)}")
        
        category_json_path = os.path.join(PROJ_DIR, f"data/mapping/{dataset_name}_category.json")
        with open(category_json_path, "r") as f:
            # 假设 json 格式为 { "label1": [idx1, idx2, ...], "label2": [...], ... }
            self.category_to_idx = json.load(f)
        self.labels = list(self.category_to_idx.keys())
        print(f"Loaded category json from {category_json_path}")
        print('Category num:', len(self.labels))
    
    def __len__(self):
        return len(self.labels) # 应当为52
    
    def __getitem__(self, index):
        # 如果使用分类加载，每个样本为一个 label 对应的所有 motion
        label = self.labels[index]
        idxs = self.category_to_idx[label]
        datasets = {}
        for key in ["motion_better", "motion_worse"]:
            data_list = [self.data[int(i)][key] for i in idxs]
            datasets[key] = torch.stack(data_list)
        for key in ["mpjpe_better", "mpjpe_worse"]:
            data_list = [self.data[int(i)][key] for i in idxs]
            datasets[key] = torch.tensor(data_list)
        datasets['label'] = label
        return datasets

class motion_pair_dataset(Dataset):
    def __init__(self, dataset_name, dataset_type):
        if dataset_name == "mdmtrain":
            motion_dataset_pth = os.path.join(PROJ_DIR, f'data/mlist_{dataset_name}_{dataset_type}.pth')
        elif dataset_name == "mdmval":
            motion_dataset_pth = os.path.join(PROJ_DIR, f'data/mlist_{dataset_name}.pth')
        print(f"Loading dataset from {motion_dataset_pth}")
        self.data = torch.load(motion_dataset_pth)
        if enable_phys:
            if dataset_name == "mdmtrain":
                mpjpe_path = os.path.join(PROJ_DIR, f'data/mpjpe/{dataset_name}_mpjpe_{dataset_type}_norm.npy')
            elif dataset_name == "mdmval":
                mpjpe_path = os.path.join(PROJ_DIR, f'data/mpjpe/{dataset_name}_mpjpe_norm.npy')
            print(f"Loading mpjpe from {mpjpe_path}")
            mpjpe = np.load(mpjpe_path)
            for i in range(len(self.data)):
                self.data[i]['mpjpe_better'] = mpjpe[i][0]
                self.data[i]['mpjpe_worse'] = mpjpe[i][1]
        print(f"Dataset {dataset_name} length: ", len(self.data))
        

    def __getitem__(self, index):
        return self.data[index] # dict, DataLoader返回包含多个tensor的dict

    def __len__(self):
        return len(self.data)


def init_seeds(seed, cuda_deterministic=True, multi_gpu=False):
    print(f'init seed {seed}, cuda_deterministic {cuda_deterministic}, multi_gpu {multi_gpu}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True


def spearmanr_loss(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


def loss_func_phys(critic, phys_score, phys_gt):
    # critic: torch (batch_size, 2), phys_score: torch (batch_size, 2), mpjpe_gt: torch (batch_size, 2)
    
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    # loss_critic_list = F.cross_entropy(critic, target, reduction='none', label_smoothing=0.1)
    loss_critic_list = F.cross_entropy(critic, target, reduction='none')
    loss_critic = torch.mean(loss_critic_list) # [64,]
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    phys_error = 0
    
    loss_phys = F.mse_loss(phys_score, phys_gt)
    loss_phys = 0.0005 * loss_phys
    phys_error = torch.abs(phys_score - phys_gt).mean()
    
    loss_total = loss_critic + loss_phys
    return loss_total, loss_critic, acc, loss_phys, phys_error

def stable_pearson(x, y, eps=1e-8):
    # 输入形状检查
    assert x.shape == y.shape, "Inputs must have the same shape"
    
    # 计算均值
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    
    # 中心化
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    # 协方差
    covariance = torch.mean(x_centered * y_centered)
    
    # 方差（添加平滑项）
    x_var = torch.mean(x_centered**2) + eps
    y_var = torch.mean(y_centered**2) + eps
    
    # 相关系数
    return covariance / (torch.sqrt(x_var) * torch.sqrt(y_var))



def metric_func_corr(critic, phys_gt):
    # critic: torch (batch_size, 2), mpjpe_gt: torch (batch_size, 2)
    # 这里的batch_size是同一条prompt对应的所有motion的数量，不是一个确定的数
    
    # NOTE: calculate Critic metric: loss_critic, acc
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_critic_list = F.cross_entropy(critic, target, reduction='none')
    loss_critic = torch.mean(loss_critic_list) # [64,]
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    
    # NOTE: calculate Physics metric: 计算critic打分和phys_gt的相关性
    critic_all = critic.view(-1).cpu().numpy()
    physics_all = phys_gt.view(-1).cpu().numpy()
    # critic越大越好，phys_gt越小越好，所以这里算出来应该是一个负相关
    spearman_corr, spearman_p = spearmanr(critic_all, physics_all)
    kendall_tau, kendall_p = kendalltau(critic_all, physics_all)
    pearson_corr, pearson_p = pearsonr(critic_all, physics_all)
    phys_error = torch.abs(critic - phys_gt).mean()
    
    return loss_critic, acc, spearman_corr, kendall_tau, pearson_corr, phys_error
        

def loss_func_corr(critic, phys_gt, loss_type="plcc", critic_coef = 1.0, phys_coef = 1.0):
    # critic: torch (batch_size, 2), mpjpe_gt: torch (batch_size, 2)
    
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_critic_list = F.cross_entropy(critic, target, reduction='none')
    loss_critic = torch.mean(loss_critic_list) # [64,]
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    phys_error = torch.tensor(0)
    # phys_gt = -phys_gt # 使得phys评分也越大越好
    
    if loss_type == "mse":
        loss_phys = F.mse_loss(critic, phys_gt)
        # loss_phys = 0.0005 * loss_phys
    
    elif loss_type == "plcc":
        # NOTE: perbatch: 将batch数据展平为(N*2,)，一整个batch计算corr
        critic_flat = critic.view(-1)
        phys_flat = phys_gt.view(-1)
        loss_phys = -torch.corrcoef(torch.stack([critic_flat, phys_flat]))[0, 1]
        
        # NOTE: perpair: better-worse两条数据计算corr
        # loss_plcc = []
        # for i in range(critic.shape[0]):
        #     if torch.std(critic[i]) < 1e-8 or torch.std(phys_gt[i]) < 1e-8:
        #         corr = 0.0  # 或者其他合理的默认值
        #         print("std too small, set corr=0")
        #     else:
        #         # 使用安全的相关系数计算
        #         corr = stable_pearson(critic[i], phys_gt[i])
        #     loss_plcc.append(corr)
            
        #     # corr = torch.corrcoef(
        #     #             torch.stack([critic[i], phys_gt[i]], dim=0) # [2, 2]
        #     #         )[0, 1]
        #     # loss_plcc.append(corr)
            
        # loss_phys = -sum(loss_plcc) / len(loss_plcc)
    
    elif loss_type == "srocc":
        # NOTE: perpair: better-worse两条数据计算corr
        loss_srocc = []
        for i in range(critic.shape[0]):
            loss_srocc.append(
                spearmanr_loss(critic[i].unsqueeze(0), phys_gt[i].unsqueeze(0))
            )
        loss_phys = -sum(loss_srocc) / len(loss_srocc)
    
    else:
        print("Wrong loss type!")
        exit(0)
    
    loss_total = critic_coef * loss_critic + phys_coef * loss_phys
    return loss_total, loss_critic, acc, loss_phys, phys_error


def metric_func(critic):
    
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = F.cross_entropy(critic, target, reduction='none')
    loss = torch.mean(loss_list)
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    
    return loss, acc, None, None

def loss_func(critic):
    
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = F.cross_entropy(critic, target, reduction='none')
    loss = torch.mean(loss_list)
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    
    return loss, loss, acc, None, None
    

def create_warmup_scheduler(warmup_type):
    if warmup_type == 'linear':
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    elif warmup_type == 'exponential':
            warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
    elif warmup_type == 'radam':
            warmup_scheduler = warmup.RAdamWarmup(optimizer)
    elif warmup_type == 'none':
            warmup_scheduler = warmup.LinearWarmup(optimizer, 1)
    return warmup_scheduler
  
def configure_optimization(model, learning_rate, lr_warmup, lr_decay):
    """Configure optimizer and learning rate schedulers"""
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    # Learning rate schedulers
    warmup_scheduler = warmup.RAdamWarmup(optimizer) if lr_warmup else None
    decay_scheduler = ExponentialLR(optimizer, gamma=0.995) if lr_decay else None
    
    return optimizer, decay_scheduler, warmup_scheduler


if __name__ == '__main__':
    args = parse_args()
    load_model_path = args.load_model
    exp_name = args.exp_name
    num_epochs = args.epoch
    save_checkpoint = args.save_checkpoint
    save_latest = args.save_latest
    lr_warmup = args.lr_warmup
    lr_decay = args.lr_decay
    big_model = args.big_model
    origin_model = args.origin_model
    enable_phys = args.enable_phys
    phys_bypass = args.phys_bypass
    loss_type = args.loss_type
    
    # Access the value of gpu_indices and convert it to a list of integers
    gpu_indices = [int(idx) for idx in args.gpu_indices.split(',')]
    gpu_number = len(gpu_indices)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # init_seeds(3407, multi_gpu=gpu_number > 1)
    
    batch_size = args.batch_size
    lr = args.learning_rate * (batch_size / 32)
    # lr = 1e-3 * (batch_size/32) # parallel changing

    print(f"training on gpu {gpu_indices}, training starting with batchsize {batch_size}, lr {lr}")
    print(f"critic_coef: {args.critic_coef}, phys_coef: {args.phys_coef}")
    # Init wandb
    if not args.debug:
        wandb.init(project="mocritic", name=exp_name, resume=False)
    
    # Instantiate your dataset
    train_loader, val_loader = create_data_loaders(args.dataset, batch_size)
    
    
    # Instantiate your model, loss function, and optimizer
    if big_model:
        model = MotionCritic(phys_bypass=phys_bypass, depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
    else:
        model = MotionCritic(phys_bypass=phys_bypass, depth=3, dim_feat=128, dim_rep=256, mlp_ratio=2)
    model = torch.nn.DataParallel(model)
    model.to(device)
    if phys_bypass:
        criterion = loss_func_phys
    elif enable_phys:
        criterion = loss_func_corr
    else:
        criterion = loss_func  # Assuming your loss_func is already defined
    
    # Create your optimizer
    optimizer, scheduler, warmup_scheduler = configure_optimization(model, lr, lr_warmup, lr_decay)
    
    start_epoch = 0
    best_accuracy = 0
    
    output_folder = f"output/{exp_name}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the model if load_model_path is provided
    if load_model_path:
        # Load the checkpoint
        checkpoint = torch.load(load_model_path)
        # Load the model and optimizer
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        best_accuracy = checkpoint['best_accuracy']
        print(f"Model loaded from {load_model_path}, starting training from epoch {start_epoch}")
    
    # Continue training from the loaded checkpoint
    for epoch in range(start_epoch, num_epochs):
        # 46740 data pairs in total; batch_size = 64; steps = 46740/64 = 730
        # log every 40 steps, 730/40 = 18 logs per epoch
        for step, batch_data in enumerate(train_loader):
            # time_start = time.time()
            # Move batch data to GPU
            # Move each tensor in the dictionary to GPU
            model.train()

            # batch_data = {key: value.cuda(device=device) for key, value in batch_data.items()} # keys: 'motion_better', 'motion_worse'
            # batch_data['motion_better'] shape: torch (batch_size, 60, 25, 3)
            # batch_data['mpjpe_better'] shape: torch (batch_size, 64)
            batch_motion = {
                key: batch_data[key].cuda(device=device)
                for key in ['motion_better', 'motion_worse'] if key in batch_data
            }
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            if phys_bypass:
                batch_mpjpe_gt = torch.stack([batch_data['mpjpe_better'], batch_data['mpjpe_worse']], dim=1).cuda(device=device)
                critic, phys_score = model(batch_motion) # torch (batch_size, 2), (batch_size, 2)
                loss, loss_critic, acc, loss_phys, phys_error = criterion(critic, phys_score, batch_mpjpe_gt)
            elif enable_phys:
                batch_mpjpe_gt = torch.stack([batch_data['mpjpe_better'], batch_data['mpjpe_worse']], dim=1).cuda(device=device)
                critic = model(batch_motion)
                loss, loss_critic, acc, loss_phys, phys_error = criterion(critic, batch_mpjpe_gt, loss_type=loss_type, critic_coef=args.critic_coef, phys_coef=args.phys_coef)
            else:
                critic = model(batch_motion)
                loss, loss_critic, acc, loss_phys, phys_error = criterion(critic)
                
            # Compute the loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if step % 40 == 0:
                # Log metrics to WandB
                if not args.debug:
                    wandb.log({
                        "Loss": loss.item(), 
                        "Loss_critic": loss_critic.item(),  
                        "Accuracy": acc.item(), 
                        "Loss_phys": loss_phys.item() if loss_phys else 0,
                        "Phys_error": phys_error.item() if phys_error else 0, 
                        "lr": optimizer.param_groups[0]['lr'],
                        })

                # Optionally, print training metrics
                print(f'Epoch {epoch + 1}, Step: {step}, Loss: {loss.item()}, Critic Loss: {loss_critic.item()}, Phys Loss: {loss_phys.item()}, Accuracy: {acc.item()}')
                # Remove batch_data from GPU to save memory

            batch_motion = {key: value.detach().cpu() for key, value in batch_motion.items()}
            if enable_phys:
                batch_mpjpe_gt = batch_mpjpe_gt.detach().cpu()
            # time_end = time.time()
            # print(f'Step Time: {time_end - time_start}')

        # evaluate the model on a epoch basis
        average_critic_loss = 0.0
        average_val_acc = 0.0
        average_phys_error = 0.0
        average_scorr = 0.0
        average_kcorr = 0.0
        average_pcorr = 0.0
        total_val_samples = 0

        if lr_decay:
            scheduler.step()
        if lr_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()

        
        with torch.no_grad():
            model.eval()
            batch_num = 0
            for batch_data in val_loader:
                batch_num += 1
                label = batch_data["label"]
                batch_motion = {
                    key: batch_data[key].squeeze(0).cuda(device=device)
                    for key in ['motion_better', 'motion_worse'] if key in batch_data
                } # {'motion_better': (batch_size, 60, 25, 3), 'motion_worse': (batch_size, 60, 25, 3)}
                batch_size = batch_motion['motion_better'].shape[0]
                print(f"Evaluating Label: {label}, batch size: {batch_size}")
                if phys_bypass:
                    batch_mpjpe_gt = torch.stack([batch_data['mpjpe_better'].squeeze(0), batch_data['mpjpe_worse'].squeeze(0)], dim=1).cuda(device=device)
                    critic, phys_score = model(batch_motion) # torch (batch_size, 2), (batch_size, 2)
                    loss, loss_critic, acc, loss_phys, phys_error = criterion(critic, phys_score, batch_mpjpe_gt)
                elif enable_phys:
                    batch_mpjpe_gt = torch.stack([batch_data['mpjpe_better'].squeeze(0), batch_data['mpjpe_worse'].squeeze(0)], dim=1).cuda(device=device)
                    critic = model(batch_motion)
                    loss_critic, acc, spearman_corr, kendall_tau, pearson_corr, phys_error = metric_func_corr(critic, batch_mpjpe_gt)
                    print(pearson_corr)
                else:
                    critic = model(batch_motion)
                    loss_critic, acc = metric_func(critic)

                total_val_samples += batch_size
                average_critic_loss += loss_critic.item() * batch_size
                average_val_acc += acc.item() * batch_size
                
                if enable_phys:
                    average_phys_error += phys_error.item() * batch_size
                    average_scorr += spearman_corr.item()
                    average_kcorr += kendall_tau.item()
                    average_pcorr += pearson_corr.item()
                    
                    batch_mpjpe_gt = batch_mpjpe_gt.detach().cpu()
                
                batch_motion = {key: value.detach().cpu() for key, value in batch_motion.items()}
                torch.cuda.empty_cache()

            average_critic_loss /= total_val_samples
            average_val_acc /= total_val_samples
            average_phys_error /= total_val_samples
            average_scorr /= batch_num
            average_kcorr /= batch_num
            average_pcorr /= batch_num
            
            metrics = {
                "val_critic_Loss": average_critic_loss,
                "val_Accuracy": average_val_acc,
                "val_Phys_error": average_phys_error,
                "val_spearman_corr": average_scorr,
                "val_kendall_corr": average_kcorr,
                "val_pearson_corr": average_pcorr
            }
            
            if not args.debug:
                wandb.log(metrics)

            # Optionally, print training metrics
            print(f'Epoch {epoch + 1}, val_critic_Loss: {average_critic_loss}, val_Accuracy: {average_val_acc}, val_Phys_error: {average_phys_error}, val_pearson_corr: {average_pcorr}')


        # Save the best model based on validation accuracy
        if average_val_acc > best_accuracy:
            best_accuracy = average_val_acc
            best_model_state = model.state_dict()
            best_checkpoint_path = f"{output_folder}/best_checkpoint.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'accuracy': acc.item(),
            }, best_checkpoint_path)
            print(f"Best model saved at {best_checkpoint_path}")

        if save_checkpoint:
            # Save checkpoint every k epochs
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = f"{output_folder}/checkpoint_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'accuracy': acc.item(),
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        if save_latest:
            # Save latest checkpoint
            checkpoint_path = f"{output_folder}/checkpoint_latest.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy,
                'accuracy': acc.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    
    

import sys
import os
PROJ_DIR = os.path.dirname(os.path.abspath(__file__)) # 'MotionCritic/MotionCritic'
sys.path.append(PROJ_DIR)
os.environ['WANDB__EXECUTABLE'] =  '/home/zhaosh/miniconda3/envs/mocritic/bin/python'
os.environ['WANDB_DIR'] = PROJ_DIR + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = PROJ_DIR + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = PROJ_DIR + '/wandb/.config/'


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

# this might be useful
# torch.manual_seed(3407)
checkpoint_interval = 40

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

    parser.add_argument('--epoch', type=int, default=200,
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
    
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed (default: 0)')



    # Parse the arguments
    return parser.parse_args()


def create_data_loaders(dataset, batch_size):
    # train_pth_name, val_pth_name = get_dataset_file(dataset)
    # train_pth = os.path.join(PROJ_DIR, 'data/'+ train_pth_name)
    # val_pth = os.path.join(PROJ_DIR, 'data/'+ val_pth_name)
    train_motion_pairs = motion_pair_dataset(dataset_name=f'mlist_mdmfull_train_{dataset}.pth')
    val_motion_pairs = motion_pair_dataset(dataset_name="mlist_mdmfull_val.pth")
    
    # Instantiate DataLoader
    train_loader = DataLoader(train_motion_pairs, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_motion_pairs, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    return train_loader, val_loader


class motion_pair_dataset(Dataset):
    def __init__(self, dataset_name):
        motion_dataset_pth = os.path.join(PROJ_DIR, 'data/'+ dataset_name)
        self.data = torch.load(motion_dataset_pth)

    def __getitem__(self, index):
        return self.data[index]

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


def loss_func(critic):
    
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = F.cross_entropy(critic, target, reduction='none')
    loss = torch.mean(loss_list)
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    
    return loss, loss_list, acc


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
    
    # Access the value of gpu_indices and convert it to a list of integers
    gpu_indices = [int(idx) for idx in args.gpu_indices.split(',')]
    gpu_number = len(gpu_indices)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    init_seeds(args.seed, multi_gpu=gpu_number > 1)
    
    batch_size = args.batch_size
    lr = args.learning_rate * (batch_size / 32)
    # lr = 1e-3 * (batch_size/32) # parallel changing

    print(f"training on gpu {gpu_indices}, training starting with batchsize {batch_size}, lr {lr}")

    # Init wandb
    if not args.debug:
        wandb.init(project="mocritic", name=exp_name, resume=False)
    
    # Instantiate your dataset
    train_loader, val_loader = create_data_loaders(args.dataset, batch_size)
    
    
    # Instantiate your model, loss function, and optimizer
    if big_model:
        model = MotionCritic(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
    else:
        model = MotionCritic(depth=3, dim_feat=128, dim_rep=256, mlp_ratio=2)
    model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = loss_func  # Assuming your loss_func is already defined
    
    # Create your optimizer
    optimizer, scheduler, warmup_scheduler = configure_optimization(model, lr, lr_warmup, lr_decay)
    
    start_epoch = 0
    best_accuracy = 0
    
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
            # Move batch data to GPU
            # Move each tensor in the dictionary to GPU
            model.train()

            batch_data = {key: value.cuda(device=device) for key, value in batch_data.items()} # keys: 'motion_better', 'motion_worse'
            # batch_data['motion_better'] shape: (batch_size, 60, 25, 3)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            critic = model(batch_data) # (batch_size, 2)

            # Compute the loss
            loss, _, acc = criterion(critic)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if step % 40 == 0:
                # Log metrics to WandB
                if not args.debug:
                    wandb.log({"Loss": loss.item(), "Accuracy": acc.item(), "lr": optimizer.param_groups[0]['lr']})

                # Optionally, print training metrics
                print(f'Epoch {epoch + 1}, Step: {step}, Loss: {loss.item()}, Accuracy: {acc.item()}')
                # Remove batch_data from GPU to save memory

            batch_data = {key: value.detach().cpu() for key, value in batch_data.items()}


        # evaluate the model on a epoch basis
        average_val_loss = 0.0
        average_val_acc = 0.0
        total_val_samples = 0

        if lr_decay:
            scheduler.step()
        if lr_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()

        
        with torch.no_grad():
            model.eval()

            for val_batch_data in val_loader:
                val_batch_data = {key: value.cuda(device=device) for key, value in val_batch_data.items()}
                val_loss, _, val_acc = criterion(model(val_batch_data))

                val_batch_size = len(val_batch_data)
                average_val_acc += val_acc.item() * val_batch_size
                average_val_loss += val_loss.item() * val_batch_size
                total_val_samples += val_batch_size
                val_batch_data = {key: value.detach().cpu() for key, value in val_batch_data.items()}
                # del val_batch_data
                torch.cuda.empty_cache()

            # Calculate average loss and accuracy
            average_val_loss = average_val_loss / total_val_samples
            average_val_acc = average_val_acc / total_val_samples
            
            if not args.debug:
                wandb.log({"val_Loss": average_val_loss, "val_Accuracy": average_val_acc})

            # Optionally, print training metrics
            print(f'Epoch {epoch + 1}, val_Loss: {average_val_loss}, val_Accuracy: {average_val_acc}')


        # Save the best model based on validation accuracy
        if average_val_acc > best_accuracy:
            best_accuracy = average_val_acc
            best_model_state = model.state_dict()
            best_checkpoint_path = f"{exp_name}_best_checkpoint.pth"
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
                checkpoint_path = f"{exp_name}_checkpoint_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'best_accuracy': best_accuracy,
                    'accuracy': acc.item(),
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        if save_latest:
            # Save latest checkpoint
            checkpoint_path = f"{exp_name}_checkpoint_latest.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'best_accuracy': best_accuracy,
                'accuracy': acc.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    
    

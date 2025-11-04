import sys
import os
import argparse
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import hashlib
import pickle
import json

# Initialize argparse
parser = argparse.ArgumentParser(description='Check for overlap between two motion datasets')

# Add command-line arguments
parser.add_argument('--dataset1', type=str, required=True,
                    help='Path to the first dataset (.pth file)')
parser.add_argument('--dataset2', type=str, required=True,
                    help='Path to the second dataset (.pth file)')
parser.add_argument('--proj_dir', type=str, default='.',
                    help='Project directory path')
parser.add_argument('--detailed', action='store_true',
                    help='Show detailed information about overlapping samples')
parser.add_argument('--method', type=str, default='fingerprint',
                    choices=['fingerprint', 'sum', 'hash', 'exact'],
                    help='Method for detecting overlaps: fingerprint (default), sum, hash, or exact')
parser.add_argument('--tolerance', type=float, default=1e-6,
                    help='Tolerance for float comparison (used with sum method)')
parser.add_argument('--key_subset', type=str, default=None,
                    help='Comma-separated list of specific keys to check (default: all keys)')

# Parse the arguments
args = parser.parse_args()

# Set up paths
PROJ_DIR = args.proj_dir
sys.path.append(PROJ_DIR)

# Process key subset if provided
key_subset = None
if args.key_subset:
    key_subset = args.key_subset.split(',')
    print(f"Will only check the following keys: {key_subset}")

# Dataset class (same as in train_reproduce.py)
class motion_pair_dataset(Dataset):
    def __init__(self, motion_pair_list_name):
        self.data = torch.load(motion_pair_list_name)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def get_data_fingerprint(data_item, method='fingerprint'):
    """
    Generate a fingerprint for a data item to compare for equality.
    Different methods are available for comparison.
    """
    if key_subset:
        # Only process specified keys
        keys_to_process = [k for k in key_subset if k in data_item]
    else:
        keys_to_process = list(data_item.keys())
    
    if method == 'fingerprint':
        # Create a string fingerprint based on tensor shapes and sums
        fingerprint = ""
        for key in keys_to_process:
            value = data_item[key]
            if isinstance(value, torch.Tensor):
                fingerprint += f"{key}:{value.shape}:{value.sum().item():.6f}_"
        return fingerprint
    
    elif method == 'sum':
        # Use a tuple of tensor sums for each key
        sums = []
        for key in keys_to_process:
            value = data_item[key]
            if isinstance(value, torch.Tensor):
                sums.append((key, float(value.sum().item())))
        return tuple(sums)
    
    elif method == 'hash':
        # Use a hash of the tensor contents
        hasher = hashlib.md5()
        for key in keys_to_process:
            value = data_item[key]
            if isinstance(value, torch.Tensor):
                # Get numpy array and update hash
                tensor_bytes = value.cpu().numpy().tobytes()
                hasher.update(tensor_bytes)
        return hasher.hexdigest()
    
    elif method == 'exact':
        # For exact comparison, return a representation of the entire item
        # This is memory intensive but ensures exact comparison
        simplified_item = {}
        for key in keys_to_process:
            value = data_item[key]
            if isinstance(value, torch.Tensor):
                simplified_item[key] = value.cpu().numpy()
        try:
            # Try to create a hash of the pickled representation
            item_bytes = pickle.dumps(simplified_item)
            return hashlib.md5(item_bytes).hexdigest()
        except:
            # Fall back to string representation if pickling fails
            return str(simplified_item)

def are_items_equal(item1, item2, tolerance=1e-6):
    """Check if two data items are exactly equal within tolerance."""
    if key_subset:
        # Only process specified keys
        keys_to_check = [k for k in key_subset if k in item1 and k in item2]
    else:
        # Check all keys that exist in both items
        keys_to_check = [k for k in item1.keys() if k in item2]
    
    if not keys_to_check:
        return False
    
    # Check each tensor for equality
    for key in keys_to_check:
        tensor1 = item1[key]
        tensor2 = item2[key]
        
        if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
            # Check shape first
            if tensor1.shape != tensor2.shape:
                return False
            
            # Check values with tolerance
            if not torch.allclose(tensor1, tensor2, rtol=tolerance, atol=tolerance):
                return False
    
    return True

                    
    

def main():
    # Load datasets
    print(f"Loading dataset 1: {args.dataset1}")
    dataset1_path = os.path.join(PROJ_DIR, 'data', args.dataset1) if not os.path.isabs(args.dataset1) else args.dataset1
    dataset1 = torch.load(dataset1_path)
    
    print(f"Loading dataset 2: {args.dataset2}")
    dataset2_path = os.path.join(PROJ_DIR, 'data', args.dataset2) if not os.path.isabs(args.dataset2) else args.dataset2
    dataset2 = torch.load(dataset2_path)
    
    print(f"Dataset 1 contains {len(dataset1)} samples")
    print(f"Dataset 2 contains {len(dataset2)} samples")
    
    # Print example of data structure
    # if len(dataset1) > 0:
    #     print("\nExample data item structure:")
    #     example_item = dataset1[0]
    #     if isinstance(example_item, dict):
    #         for key, value in example_item.items():
    #             if isinstance(value, torch.Tensor):
    #                 print(f"  {key}: Tensor shape {value.shape}, dtype {value.dtype}")
    #             else:
    #                 print(f"  {key}: {type(value)}")
    
    # Generate fingerprints for all items in dataset1
    print(f"\nGenerating fingerprints for dataset 1 using method: {args.method}...")
    fingerprints1 = {}
    for i in tqdm(range(len(dataset1))):
        fingerprint = get_data_fingerprint(dataset1[i], method=args.method)
        # Handle collisions by keeping a list of indices
        if fingerprint not in fingerprints1:
            fingerprints1[fingerprint] = [i]
        else:
            fingerprints1[fingerprint].append(i)
    
    # Check for overlaps
    print("Checking for overlaps...")
    overlaps = []
    
    for i in tqdm(range(len(dataset2))):
        fingerprint = get_data_fingerprint(dataset2[i], method=args.method)
        if fingerprint in fingerprints1:
            for idx1 in fingerprints1[fingerprint]:
                overlaps.append((i, idx1))
    
    # Report results
    print(f"\nFound {len(overlaps)} potential overlapping samples between the datasets")
    print(f"Potential overlap percentage: {len(overlaps)/len(dataset1)*100:.2f}% of dataset1")
    print(f"Potential overlap percentage: {len(overlaps)/len(dataset2)*100:.2f}% of dataset2")
    
    # Show detailed information if requested
    if args.detailed and overlaps:
        print("\nDetailed overlap information:")
        for idx2, idx1 in overlaps[:10]:  # Limit to first 10 for brevity
            print(f"Dataset2[{idx2}] potentially overlaps with Dataset1[{idx1}]")
        
        if len(overlaps) > 10:
            print(f"... and {len(overlaps)-10} more")

    # Check for exact equality (slow but thorough)
    if overlaps:
        print("\nPerforming exact equality check on found overlaps...")
        exact_matches = []
        
        for idx2, idx1 in tqdm(overlaps):
            item1 = dataset1[idx1]
            item2 = dataset2[idx2]
            
            if are_items_equal(item1, item2, tolerance=args.tolerance):
                exact_matches.append((idx2, idx1))
        
        print(f"Exact matches: {len(exact_matches)} out of {len(overlaps)} potential overlaps")
        
        # Report exact match details
        if args.detailed and exact_matches:
            print("\nExact match details:")
            for idx2, idx1 in exact_matches[:10]:  # Limit to first 10 for brevity
                print(f"Dataset2[{idx2}] exactly matches Dataset1[{idx1}]")
            
            if len(exact_matches) > 10:
                print(f"... and {len(exact_matches)-10} more")
    
    # Save the results to a file
    results_file = "data/mapping/flame_compare.json"
    with open(results_file, 'w') as f:
        json.dump(exact_matches, f) # [(idx_new, idx_old)...]
    
    # results_file = f"overlap_results_{os.path.basename(args.dataset1)}_{os.path.basename(args.dataset2)}.txt"
    # with open(results_file, 'w') as f:
    #     f.write(f"Dataset 1: {args.dataset1}, size: {len(dataset1)}\n")
    #     f.write(f"Dataset 2: {args.dataset2}, size: {len(dataset2)}\n")
    #     f.write(f"Comparison method: {args.method}\n")
    #     f.write(f"Total potential overlaps: {len(overlaps)}\n")
        
    #     if overlaps:
    #         # f.write(f"Exact matches: {len(exact_matches)}\n")
    #         # f.write(f"Overlap percentage: {len(exact_matches)/len(dataset1)*100:.2f}% of dataset1\n")
    #         # f.write(f"Overlap percentage: {len(exact_matches)/len(dataset2)*100:.2f}% of dataset2\n")
            
    #         if exact_matches:
    #             f.write("\nExact match indices (dataset2_idx, dataset1_idx):\n")
    #             for idx2, idx1 in exact_matches:
    #                 f.write(f"{idx2}, {idx1}\n")
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main() 

# 
# python dataset_checker.py --dataset1 motion_dataset/mlist_mdmfull_trainshuffle.pth --dataset2 motion_dataset/mlist_mdmfull_train_corrected.pth --method hash --detailed
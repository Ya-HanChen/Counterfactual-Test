import torch
import glob
import os
import re

# --- Configuration ---
# 1. Directory containing the .pt files
# (Ensure this matches the --save_hidden argument in create_text.py)
HIDDEN_DIR = r"C:\Users\yachun\Desktop\阿\hidden_states_十屆全文本"

# 2. File pattern to match
# (Based on your script, it should be 'hidden_last_*.pt')
FILE_PATTERN = "hidden_last_*.pt"

# 3. Output path for the merged file
OUTPUT_FILE = r"C:\Users\yachun\Desktop\RR_RR\synthetic_data\6th_LY\hidden_states_combined_全文本_10th.pt"
# --- End Configuration ---

def get_file_number(filename):
    """
    Extracts the number from the filename (e.g., '49' from 'hidden_last_49.pt') for sorting.
    """
    match = re.search(r'hidden_last_(\d+)\.pt', filename)
    if match:
        return int(match.group(1))
    return -1

def merge_tensors():
    """
    Finds all matching .pt files, sorts them numerically, and merges them.
    """
    print(f"[*] Searching directory: {HIDDEN_DIR}")
    print(f"[*] File pattern: {FILE_PATTERN}")

    # 1. Find all matching files
    search_path = os.path.join(HIDDEN_DIR, FILE_PATTERN)
    file_list = glob.glob(search_path)

    if not file_list:
        print(f"[!] Error: No files found in {search_path}.")
        print("    Please check HIDDEN_DIR and FILE_PATTERN.")
        return

    # 2. Sort files numerically (0, 1, 2...)
    # Crucial step: ensures files are merged in the correct order.
    file_list.sort(key=lambda f: get_file_number(os.path.basename(f)))
    
    print(f"[*] Found {len(file_list)} files. Loading in order...")
    
    tensors_to_merge = []
    for file_path in file_list:
        try:
            # Load single tensor (CPU)
            tensor = torch.load(file_path, map_location='cpu') 
            tensors_to_merge.append(tensor)
        except Exception as e:
            print(f"[!] Error loading file {file_path}: {e}")
            return

    # 3. Concatenate all tensors along dim 0
    # Merges N tensors of shape [1, 4096] into one [N, 4096] tensor.
    try:
        combined_tensor = torch.cat(tensors_to_merge, dim=0)
    except Exception as e:
        print(f"[!] Error merging tensors: {e}")
        print("    Ensure all .pt files have the same dimensions.")
        return

    # 4. Save the combined tensor
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True) 
        torch.save(combined_tensor, OUTPUT_FILE)
        
        print("\n" + "="*30)
        print(f"[✓] Success!")
        print(f"    Merged tensor saved to: {OUTPUT_FILE}")
        print(f"    Final Shape: {combined_tensor.shape}")
        print("="*30)
        
    except Exception as e:
        print(f"[!] Error saving file {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    merge_tensors()
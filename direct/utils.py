import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, ConcatDataset
from collections import defaultdict
import pandas as pd

def create_eval_dataset(base_input_dir, noise_var, rain_len, N, p):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset paths
    dataset_path_00 = os.path.join(base_input_dir, "noise_0.00_rain_00")
    dataset_path_selected = os.path.join(base_input_dir, f"noise_{noise_var:.2f}_rain_{rain_len:02d}")

    if not os.path.exists(dataset_path_00) or not os.path.exists(dataset_path_selected):
        print(f"One of the dataset paths does not exist: {dataset_path_00} or {dataset_path_selected}, skipping...")
        return None

    # Load datasets
    dataset_00 = ImageFolder(root=dataset_path_00, transform=transform)
    dataset_selected = ImageFolder(root=dataset_path_selected, transform=transform)

    # Determine number of images to sample from each dataset
    N_00 = int(N * p)
    N_selected = N - N_00

    per_class_00 = N_00 // 10
    per_class_selected = N_selected // 10

    # Organize images by class
    class_indices_00 = defaultdict(list)
    class_indices_selected = defaultdict(list)

    for idx, (_, label) in enumerate(dataset_00.samples):
        class_indices_00[label].append(idx)

    for idx, (_, label) in enumerate(dataset_selected.samples):
        class_indices_selected[label].append(idx)

    subset_indices_00 = []
    subset_indices_selected = []

    for class_idx in range(10):
        subset_indices_00.extend(class_indices_00[class_idx][:per_class_00])
        subset_indices_selected.extend(class_indices_selected[class_idx][:per_class_selected])

    # Create subsets from each dataset
    subset_00 = Subset(dataset_00, subset_indices_00)
    subset_selected = Subset(dataset_selected, subset_indices_selected)

    # Combine the two subsets using ConcatDataset
    combined_dataset = ConcatDataset([subset_00, subset_selected])

    return combined_dataset


def compute_cdf(class_probabilities):
    p_values = np.arange(0, 1.001, 0.001)
    cdfs = []
    
    for class_id in range(10):
        if len(class_probabilities[class_id]) > 0:
            data_sorted = np.sort(class_probabilities[class_id])
            cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            cdf_interp = np.interp(p_values, data_sorted, cdf, left=0, right=1)
            cdfs.append(np.round(cdf_interp, 3))
        else:
            cdfs.append(np.zeros_like(p_values))
    
    return cdfs

def compute_p_cmm_test(cdfs, thresholds):
    cdf_at_thresholds = []

    for _, threshold_row in thresholds.iterrows():
        class_id = threshold_row['Class']
        threshold = threshold_row['Threshold']

        cdf_values = cdfs[int(class_id)]
        cdf_value_at_threshold = round(np.interp(threshold, np.arange(0, 1.001, 0.001), cdf_values), 3)     
        cdf_at_thresholds.append([class_id, threshold_row['Noise_Variance'], int(threshold_row['Rain_Length']), cdf_value_at_threshold])

    return pd.DataFrame(cdf_at_thresholds, columns=["Class", "Threshold_Noise_Variance", "Threshold_Rain_Length", "CDF"])

def fill_table_test(p_cmm_test):
    class_range = range(10)
    noise_variance_range = np.arange(0, 0.21, 0.01)
    rain_length_range = range(21)

    complete_data = pd.DataFrame(
        [(c, nv, rl) for c in class_range for nv in noise_variance_range for rl in rain_length_range],
        columns=['Class', 'Threshold_Noise_Variance', 'Threshold_Rain_Length']
    )

    merged_df = pd.merge(complete_data, p_cmm_test, how='left', on=['Class', 'Threshold_Noise_Variance', 'Threshold_Rain_Length'])
    merged_df['CDF'] = merged_df['CDF'].fillna(0)

    return merged_df

def predict_noise(p_cmm_test, p_cmm, noise_var, rain_len):
    noise_var = round(noise_var, 2)
    rain_len = int(rain_len)

    p_cmm_group = p_cmm[(p_cmm['Noise_Variance'].round(2) == noise_var) & (p_cmm['Rain_Length'] == rain_len)].sort_values(by='Class')

    min_diff = float('inf')
    best_match = None

    for (threshold_noise_var, threshold_rain_len), result_group in p_cmm_test.groupby(['Threshold_Noise_Variance', 'Threshold_Rain_Length']):
        diff = np.sum(np.abs(result_group['CDF'].values - p_cmm_group['CDF'].values))
        
        if diff < min_diff:
            min_diff = diff
            best_match = (threshold_noise_var, threshold_rain_len)

    return best_match

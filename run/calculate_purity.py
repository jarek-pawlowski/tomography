import sys

from src.model_utils import calculate_mean_model_output_with_varied_feature
from src.test_model import test_output_statistics_for_given_feature, test_output_statistics_varying_feature, test_varying_feature_with_value_range
sys.path.append('./')
from copy import deepcopy
import pickle
import numpy as np

import matplotlib.pyplot as plt
from seaborn import heatmap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import DensityMatrixDataset
from src.data_utils import calculate_states_count
from src.tomography_utils_torch import calculate_concurrence_from_measurements
from src.log import log_metrics_to_file, plot_metrics_from_file, plot_metrics_from_files

batch_size = 512
test_dataset = DensityMatrixDataset(root_path='./data/val/')
# test_dataset = VectorDensityMatrixDataset(root_path='./data/val/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

plot_path = f'./plots/2qbits_purity_distribution_ent.png'

device = torch.device('cpu')

def purity(data):
    complex_tensor = data[:, 0] + 1j * data[:, 1]
    matrix_square = torch.linalg.matrix_power(complex_tensor, 2)
    trace = torch.vmap(torch.trace)(matrix_square)
    return torch.real(trace)

def is_pure(data, min_purity=0.990, max_purity=1.000):
    p = torch.abs(purity(data))
    return (p > min_purity) & (p < max_purity)

def is_in_range(label):
    return (label > 0.99)
    # return (label < 1.e-6)


thresholds = [
    [-0.01, 0.1],
    [0.1, 0.2],
    [0.2, 0.3],
    [0.3, 0.4],
    [0.4, 0.5],
    [0.5, 0.6],
    [0.6, 0.7],
    [0.7, 0.8],
    [0.8, 0.9],
    [0.9, 1.01]
]

data = []

save_path = "./purity_distribution_ent.pickle"
try:
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
        print(f"Data loaded from {save_path}")
except FileNotFoundError:
    for th_0, th_1 in thresholds:
        print(f"Threshold: {th_0}")
        is_purity_in_range = lambda x: is_pure(x, th_0, th_1)
        n_pure = calculate_states_count(test_loader, data_filter=is_purity_in_range, label_filter=is_in_range)
        print(f"Number of pure states: {n_pure}")
        data.append(((th_0 + th_1) / 2, n_pure))
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
        print(f"Data saved to {save_path}")

# Plotting the data
plt.figure(figsize=(10, 6))
plt.title("Purity Distribution of Entangled States")
plt.xlabel("Purity Threshold")
plt.ylabel("Number of Entangled Pure States")

normalized_data = [(x[0], x[1] / sum([x[1] for x in data])) for x in data]

plt.plot([x[0] for x in normalized_data], [x[1] for x in normalized_data], marker='o', linestyle='-')

plt.savefig(plot_path)

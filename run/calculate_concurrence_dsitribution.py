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

from src.datasets import MeasurementDataset
from src.data_utils import calculate_dataset_histogram
from src.tomography_utils_torch import calculate_concurrence_from_measurements
from src.log import log_metrics_to_file, plot_metrics_from_file, plot_metrics_from_files

batch_size = 512
test_dataset = MeasurementDataset(root_path='./data/val/')
# test_dataset = VectorDensityMatrixDataset(root_path='./data/val/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

plot_path = f'./plots/2qbits_concurrence_binary_distribution.png'

device = torch.device('cpu')

bins = torch.tensor([-0.1, 1.e-6, 1.1])
histogram = calculate_dataset_histogram(test_loader, device, bins=bins)
print(histogram)
plt.bar(range(2), histogram / histogram.sum())
plt.xticks(range(2), ['0', '1'])
plt.savefig(plot_path)

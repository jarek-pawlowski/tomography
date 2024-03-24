import sys
sys.path.append('./')

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.torch_measure import test_reconstruction_measurement_noise_for_variance
from src.torch_utils import torch_bures_distance
from src.logging import log_metrics_to_file, plot_metrics_from_file

batch_size = 512
test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

results_path = './logs/rho_varying_multiple_measurements/rho_test_varying_measurement_clipped_optimized.log'
plot_path = './plots/rho_varying_multiple_measurements/rho_test_varying_measurement_clipped_optimized.png'

num_measurements = 16
num_repetitions = 10
variance = 1.

rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y, reduction='none'))
mse_loss = nn.MSELoss(reduction='none')
bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='none')
criterions = {
    'test_rmse_loss': rmse_loss,
    'test_mse_loss': mse_loss,
    'bures_distance': bures_distance
}

strategy = 'optimized_tomography'
method = 'MLE'
use_intensity = False

for i in range(6, num_measurements + 1):
    print('Num measurements:', i)
    avg_metrics = {}
    for j in tqdm(range(num_repetitions), desc='Averaging metrics...'):
        measurement_ids = np.random.randint(0, num_measurements, i) # maybe better to repeat for all measurements (at least for i = 1)
        test_metrics = test_reconstruction_measurement_noise_for_variance(test_loader, criterions, varying_input_idx=measurement_ids, variance=1., strategy=strategy, method=method, use_intensity=use_intensity)
        for metric_name, metric_value in test_metrics.items():
            if metric_name not in avg_metrics:
                avg_metrics[metric_name] = 0
            avg_metrics[metric_name] += metric_value / num_repetitions
    write_mode = 'w' if i == 1 else 'a'
    log_metrics_to_file(avg_metrics, results_path, write_mode=write_mode, xaxis=i, xaxis_name='num measurements')
plot_metrics_from_file(results_path, title=f'Metrics for tomography reconstruction', save_path=plot_path, xaxis='num measurements')

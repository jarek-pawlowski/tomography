import sys
sys.path.append('./')
from itertools import combinations

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.torch_measure import test_reconstruction_measurement_noise_for_variance
from src.torch_utils import torch_bures_distance
from src.logging import log_metrics_to_file, plot_metrics_from_file
from src.utils_measure import Kwiat


def list_to_str(l):
    return '_'.join([str(x) for x in l])

batch_size = 512
num_qubits = 2
test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True, num_qubits=num_qubits)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

results_path = './logs/2qbit/mle_rho_varying_multiple_measurements/rho_test_varying_measurement_clipped_tomography{}.log'
avg_results_path = './logs/2qbit/mle_rho_test_varying_measurement_clipped_tomography_avg.log'
plot_path = './plots/2qbit/mle_rho_varying_multiple_measurements/rho_test_varying_measurement_clipped_tomography.png'

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

for i in range(1, num_measurements):
    print('Num measurements:', i)
    avg_metrics = {}
    used_measuerements = []
    num_possible_measurements = len(list(combinations(range(len(Kwiat.basis)**num_qubits), i))) if i > 0 else 1
    num_repetitions_i = min(num_possible_measurements, num_repetitions)
    write_mode = 'w' if i == 0 else 'a'
    for j in tqdm(range(num_repetitions_i), desc='Averaging metrics...'):
        if i > 0:
            while (measurement_ids := set(np.random.choice(num_measurements, i, replace=False))) in used_measuerements: # maybe better to repeat for all measurements (at least for i = 1)
                pass
            used_measuerements.append(measurement_ids)
            measurement_ids = list(measurement_ids)
        else:
            measurement_ids = None
        test_metrics = test_reconstruction_measurement_noise_for_variance(test_loader, criterions, varying_input_idx=measurement_ids, variance=1., strategy=strategy, method=method, use_intensity=use_intensity)
        if i > 0:
            log_metrics_to_file(test_metrics, results_path.format(f'_m{list_to_str(measurement_ids)}'), write_mode=write_mode, xaxis=i, xaxis_name='num measurements')
        for metric_name, metric_value in test_metrics.items():
            if metric_name not in avg_metrics:
                avg_metrics[metric_name] = 0
            avg_metrics[metric_name] += metric_value / num_repetitions_i
    log_metrics_to_file(avg_metrics, avg_results_path, write_mode=write_mode, xaxis=i, xaxis_name='num measurements')
plot_metrics_from_file(avg_results_path, title=f'Metrics for tomography reconstruction', save_path=plot_path, xaxis='num measurements')

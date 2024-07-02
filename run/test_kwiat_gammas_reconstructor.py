import sys
sys.path.append('./')
import os

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.model import GammasReconstructor
from src.torch_utils import test_kwiat_gammas_reconstruction, torch_bures_distance
from src.logging import log_metrics_to_file, plot_metrics_from_file
from src.utils_measure import Kwiat

def list_to_str(l):
    return '_'.join([str(x) for x in l])

    
def calculate_single_run_metrics(test_loader: DataLoader, measurement_subset_len: int):
    num_qubits = 2
    measurement_subset = random.sample(range(len(Kwiat.basis)**num_qubits), measurement_subset_len)

    criterion = nn.MSELoss()
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')
    criterions = {
        'test_loss': criterion,
        'bures_distance': bures_distance
    }
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    test_metrics = test_kwiat_gammas_reconstruction(device, test_loader, criterions, measurements_subset=measurement_subset)
    best_test_loss = test_metrics['test_loss']
    best_bures_distance = test_metrics['bures_distance']
    return best_test_loss, best_bures_distance


if __name__ == '__main__':
    num_repetitions = 100
    min_num_measurements = 16
    max_num_measurements = 16
    log_path = f'./logs/density_matrix_reconstructor_from_basis_gammas_measurements_subset16.log'

    batch_size = 64
    test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for num_measurements in range(min_num_measurements, max_num_measurements + 1):
        print(f'Running for {num_measurements} measurements')
        metrics = {
            'test_loss_avg': 0,
            'test_loss_min': float('inf'),
            'test_loss_max': 0,
            'bures_distance_avg': 0,
            'bures_distance_min': 1.,
            'bures_distance_max': 0,
            'successes_ratio': 0.
        }
        num_successes = 0

        for _ in range(num_repetitions):
            try:
                loss, bures_distance = calculate_single_run_metrics(test_loader, num_measurements)
                metrics['test_loss_avg'] += loss
                metrics['test_loss_min'] = min(metrics['test_loss_min'], loss)
                metrics['test_loss_max'] = max(metrics['test_loss_max'], loss)
                metrics['bures_distance_avg'] += bures_distance
                metrics['bures_distance_min'] = min(metrics['bures_distance_min'], bures_distance)
                metrics['bures_distance_max'] = max(metrics['bures_distance_max'], bures_distance)
                num_successes += 1
            except:
                pass

        metrics['successes_ratio'] = num_successes / num_repetitions
        print('Successes ratio:', metrics['successes_ratio'])
        if num_successes > 0:
            metrics['test_loss_avg'] /= num_successes
            metrics['bures_distance_avg'] /= num_successes

        write_mode = 'w' if num_measurements == min_num_measurements else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=num_measurements, xaxis_name='num_measurements')
    plot_metrics_from_file(log_path, title='Metrics', save_path=f'./plots/density_matrix_reconstructor_from_basis_gammas_measurements_subset16_metrics.png', xaxis='num_measurements')
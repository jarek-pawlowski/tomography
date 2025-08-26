import sys

import numpy as np

from src.test_heuristics import test_kwiat_gammas_reconstruction
sys.path.append('./')
import os

from math import comb
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.model import GammasReconstructor
from src.criterions import torch_bures_distance, complex_distance_matrix_elements_avg
from src.log import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat


def list_to_str(l):
    return '_'.join([str(x) for x in l])

def convert_tensor_to_dict(array: torch.Tensor):
    result = {}
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            result[f'{i}{j}'] = array[i, j].item()
    return result
    
def calculate_single_run_metrics(dir_name: str, test_loader: DataLoader, measurement_subset_len: int, inverse: str, enforce_valid_density_matrix: bool, num_qubits: int = 2, previously_used_measurements: list = []):
    while (measurement_subset := set(random.sample(range(len(Kwiat.basis)**num_qubits), measurement_subset_len))) in previously_used_measurements:
        pass
    previously_used_measurements.append(measurement_subset)

    criterion = nn.MSELoss()
    rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y, reduction='mean'))
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')
    criterions = {
        'test_mse_loss': criterion,
        'test_rmse_loss': rmse_loss,
        'bures_distance': bures_distance,
        'avg_complex_distance': complex_distance_matrix_elements_avg
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_metrics = test_kwiat_gammas_reconstruction(device, test_loader, criterions, measurements_subset=list(measurement_subset), inverse=inverse, enforce_valid_density_matrix=enforce_valid_density_matrix)
    matrix_elements_complex_distance = test_metrics.pop('avg_complex_distance')
    best_mse_loss = test_metrics['test_mse_loss']
    best_bures_distance = test_metrics['bures_distance']
    best_rmse_loss = test_metrics['test_rmse_loss']
    log_path = os.path.join(dir_name, f'reconstruction_from_m{list_to_str(measurement_subset)}.log')
    log_metrics_to_file(test_metrics, log_path, xaxis=num_measurements, xaxis_name='num_measurements')
    matrix_elements_log_path = os.path.join(dir_name, f'complex_distance_reconstruction_from_m{list_to_str(measurement_subset)}.log')
    # matrix_metrics_dict = {
    #     '00': matrix_elements_complex_distance[0, 0].item(),
    #     '01': matrix_elements_complex_distance[0, 1].item(),
    #     '10': matrix_elements_complex_distance[1, 0].item(),
    #     '11': matrix_elements_complex_distance[1, 1].item()
    # }
    matrix_metrics_dict = convert_tensor_to_dict(matrix_elements_complex_distance)
    log_metrics_to_file(matrix_metrics_dict, matrix_elements_log_path, xaxis=num_measurements, xaxis_name='num_measurements')
    return best_mse_loss, best_rmse_loss, best_bures_distance


if __name__ == '__main__':
    num_repetitions = 10
    min_num_measurements = 1
    max_num_measurements = 256
    step = 4
    num_measurements_range = np.arange(min_num_measurements, max_num_measurements - 2, step)
    num_measurements_range = np.append(num_measurements_range, [max_num_measurements - 1, max_num_measurements])

    num_qubits = 4
    inverse = 'pinv'
    enforce_valid_density_matrix = False
    dir_name = f'./logs/4qbits/density_matrix_reconstructor_from_pinv_gammas/'
    log_path = f'./logs/4qbits/density_matrix_reconstructor_from_pinv_gammas.log'

    batch_size = 64
    test_dataset = MeasurementDataset(root_path='./data/4qbits/val/', return_density_matrix=True, num_qubits=num_qubits)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for num_measurements in num_measurements_range:
        print(f'Running for {num_measurements} measurements')
        metrics = {
            'mse_loss_avg': 0,
            'mse_loss_min': float('inf'),
            'mse_loss_max': 0,
            'rmse_loss_avg': 0,
            'rmse_loss_min': float('inf'),
            'rmse_loss_max': 0,
            'bures_distance_avg': 0,
            'bures_distance_min': 1.,
            'bures_distance_max': 0,
            'successes_ratio': 0.
        }
        num_successes = 0

        used_measurements = []
        num_possible_measurements = comb(len(Kwiat.basis)**num_qubits, num_measurements)
        for _ in range(num_repetitions):
            if len(used_measurements) == num_possible_measurements:
                break
            try:
                mse_loss, rmse_loss, bures_distance = calculate_single_run_metrics(dir_name, test_loader, num_measurements, inverse, enforce_valid_density_matrix, num_qubits=num_qubits, previously_used_measurements=used_measurements)
                metrics['mse_loss_avg'] += mse_loss
                metrics['mse_loss_min'] = min(metrics['mse_loss_min'], mse_loss)
                metrics['mse_loss_max'] = max(metrics['mse_loss_max'], mse_loss)
                metrics['rmse_loss_avg'] += rmse_loss
                metrics['rmse_loss_min'] = min(metrics['rmse_loss_min'], rmse_loss)
                metrics['rmse_loss_max'] = max(metrics['rmse_loss_max'], rmse_loss)
                metrics['bures_distance_avg'] += bures_distance
                metrics['bures_distance_min'] = min(metrics['bures_distance_min'], bures_distance)
                metrics['bures_distance_max'] = max(metrics['bures_distance_max'], bures_distance)
                num_successes += 1
            except Exception as e:
                pass

        denominator = min(num_repetitions, num_possible_measurements)
        metrics['successes_ratio'] = num_successes / denominator
        print('Successes ratio:', metrics['successes_ratio'])
        if num_successes > 0:
            metrics['mse_loss_avg'] /= num_successes
            metrics['rmse_loss_avg'] /= num_successes
            metrics['bures_distance_avg'] /= num_successes

        write_mode = 'a' if num_measurements == min_num_measurements else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=num_measurements, xaxis_name='num_measurements')
    plot_metrics_from_file(log_path, title='Metrics', save_path=f'./plots/4qbits/density_matrix_reconstructor_from_pinv_gammas_metrics.png', xaxis='num_measurements')
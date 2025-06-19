import sys
sys.path.append('./')
import os
from itertools import combinations

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets import MeasurementDataset
from src.data_utils import generate_mean_sample
from src.criterions import torch_bures_distance, complex_distance_matrix_elements_avg
from src.log import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat
from src.test_heuristics import test_mean_reconstruction


def list_to_str(l):
    return '_'.join([str(x) for x in l])

def convert_tensor_to_dict(array: torch.Tensor):
    result = {}
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            result[f'{i}{j}'] = array[i, j].item()
    return result

    
def calculate_metrics(dir_name: str, test_loader: DataLoader, mean_rho: torch.Tensor):
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

    test_metrics = test_mean_reconstruction(mean_rho, test_loader, criterions, device=device)
    matrix_elements_complex_distance = test_metrics.pop('avg_complex_distance')
    matrix_elements_log_path = os.path.join(dir_name, f'complex_distance_reconstruction.log')
    matrix_metrics_dict = convert_tensor_to_dict(matrix_elements_complex_distance)
    log_metrics_to_file(matrix_metrics_dict, matrix_elements_log_path)
    return test_metrics


if __name__ == '__main__':
    num_qubits = 2
    inverse = 'pinv'
    enforce_valid_density_matrix = False
    dir_name = f'./logs/{num_qubits}qbit/density_matrix_reconstructor_from_mean/'
    log_path = f'./logs/{num_qubits}qbit/density_matrix_reconstructor_from_mean.log'

    batch_size = 64
    train_dataset = MeasurementDataset(root_path='./data/train/', return_density_matrix=True, num_qubits=num_qubits)
    train_subset = Subset(train_dataset, random.sample(range(len(train_dataset)), 10000))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True, num_qubits=num_qubits)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    mean_rho = generate_mean_sample(train_loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    metrics = calculate_metrics(dir_name, test_loader, mean_rho)    
    log_metrics_to_file(metrics, log_path)

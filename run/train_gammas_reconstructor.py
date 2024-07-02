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
from src.torch_utils import train_gammas_reconstructor, test_gammas_reconstructor, torch_bures_distance
from src.logging import log_metrics_to_file, plot_metrics_from_file
from src.utils_measure import Kwiat

def list_to_str(l):
    return '_'.join([str(x) for x in l])

    
def calculate_single_run_metrics(train_loader: DataLoader, test_loader: DataLoader, measurement_subset_len: int, dir_name: str, model_input_info: str):
    num_qubits = 2
    measurement_subset = random.sample(range(len(Kwiat.basis)**num_qubits), measurement_subset_len)
    if model_input_info == 'full':
        input_dim = num_qubits*2*2*2 + 1
    elif model_input_info == 'measurement':
        input_dim = 1
    elif model_input_info == 'measurement_basis':
        input_dim = num_qubits*2*2*2
    
    model_params = {
        'input_dim': measurement_subset_len*input_dim,
        'num_qubits': num_qubits, # 'Kwiat' basis matrices
        'layers': 6,
        'hidden_size': 64,
        'num_gammas': measurement_subset_len
    }
    model = GammasReconstructor(**model_params)

    model_name = 'mlp_density_matrix_reconstructor'
    model_name = f'{model_name}_m{list_to_str(measurement_subset)}'
    model_save_path = f'./models/{dir_name}/{model_name}.pt'

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # train & test model
    log_path = f'./logs/{dir_name}/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    num_epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')
    criterions = {
        'test_loss': criterion,
        'bures_distance': bures_distance
    }
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    best_test_loss = float('inf')
    best_bures_distance = 1.
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_gammas_reconstructor(model, device, train_loader, optimizer, epoch, criterion=criterion, log_interval=10, measurements_subset=measurement_subset, model_input_info=model_input_info)
        test_metrics = test_gammas_reconstructor(model, device, test_loader, criterions, measurements_subset=measurement_subset, model_input_info=model_input_info)
        if test_metrics['test_loss'] < best_test_loss:
            best_test_loss = test_metrics['test_loss']
            best_bures_distance = test_metrics['bures_distance']
            model.save(model_save_path)
        # make test_metrics flat
        metrics = {**train_metrics, **test_metrics}
        write_mode = 'w' if epoch == 1 else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=epoch)
    plot_metrics_from_file(log_path, title='Loss', save_path=f'./plots/{dir_name}/{model_name}_loss.png')
    return best_test_loss, best_bures_distance


if __name__ == '__main__':
    num_repetitions = 10
    min_num_measurements = 1
    max_num_measurements = 16
    model_input_info = 'measurement_basis'
    log_path = f'./logs/density_matrix_reconstructor_based_on_basis_only_from_gammas_measurements_subset.log'

    batch_size = 64
    train_dataset = MeasurementDataset(root_path='./data/train/', return_density_matrix=True)
    test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for num_measurements in range(min_num_measurements, max_num_measurements + 1):
        print(f'Running for {num_measurements} measurements')
        dir_name = f'density_matrix_reconstructor_based_on_basis_only_from_gammas_m{num_measurements}'
        metrics = {
            'test_loss_avg': 0,
            'test_loss_min': float('inf'),
            'test_loss_max': 0,
            'bures_distance_avg': 0,
            'bures_distance_min': 1.,
            'bures_distance_max': 0,
        }

        for _ in range(num_repetitions):
            loss, bures_distance = calculate_single_run_metrics(train_loader, test_loader, num_measurements, dir_name, model_input_info)
            metrics['test_loss_avg'] += loss
            metrics['test_loss_min'] = min(metrics['test_loss_min'], loss)
            metrics['test_loss_max'] = max(metrics['test_loss_max'], loss)
            metrics['bures_distance_avg'] += bures_distance
            metrics['bures_distance_min'] = min(metrics['bures_distance_min'], bures_distance)
            metrics['bures_distance_max'] = max(metrics['bures_distance_max'], bures_distance)

        metrics['test_loss_avg'] /= num_repetitions
        metrics['bures_distance_avg'] /= num_repetitions
        write_mode = 'w' if num_measurements == min_num_measurements else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=num_measurements, xaxis_name='num_measurements')
    plot_metrics_from_file(log_path, title='Metrics', save_path=f'./plots/density_matrix_reconstructor_based_on_basis_only_from_gammas_measurements_subset_metrics.png', xaxis='num_measurements')
import sys
sys.path.append('./')
import os
from itertools import combinations

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.model import TomographyCorrectionsPredictor
from src.criterions import torch_bures_distance, complex_distance_matrix_elements_avg
from src.logging import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat
from src.test_model import test_tomography_corrections_predictor
from src.train import train_tomography_corrections_predictor

def list_to_str(l):
    return '_'.join([str(x) for x in l])

    
def calculate_single_run_metrics(train_loader: DataLoader, test_loader: DataLoader, measurement_subset_len: int, dir_name: str, model_input_info: str, num_qubits:int = 2, previously_used_measurements: list = []):
    # measurement_subset = random.sample(range(len(Kwiat.basis)**num_qubits), measurement_subset_len)
    while (measurement_subset := set(random.sample(range(len(Kwiat.basis)**num_qubits), measurement_subset_len))) in previously_used_measurements:
        pass
    previously_used_measurements.append(measurement_subset)

    if model_input_info == 'full':
        input_dim = num_qubits*2*2*2 + 1
    elif model_input_info == 'measurement':
        input_dim = 1
    elif model_input_info == 'measurement_basis':
        input_dim = num_qubits*2*2*2
    
    model_params = {
        'input_dim': measurement_subset_len*input_dim,
        'num_measurements': measurement_subset_len,
        'num_gammas': 4,
        'layers': 6,
        'hidden_size': 64,
    }
    model = TomographyCorrectionsPredictor(**model_params)

    model_name = 'mlp_tomography_corrections_predictor'
    model_name = f'{model_name}_m{list_to_str(measurement_subset)}'
    model_save_path = f'./models/{dir_name}/{model_name}.pt'

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # train & test model
    log_path = f'./logs/{dir_name}/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    matrix_elements_log_path = f'./logs/{dir_name}/{model_name}_complex_distance.log'

    num_epochs = 5
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')
    criterions = {
        'test_loss': criterion,
        'bures_distance': bures_distance,
        'avg_complex_distance': complex_distance_matrix_elements_avg
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_test_loss = float('inf')
    best_bures_distance = 1.
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_tomography_corrections_predictor(model, device, train_loader, optimizer, epoch, criterion=criterion, log_interval=10, measurements_subset=list(measurement_subset), model_input_info=model_input_info)
        test_metrics = test_tomography_corrections_predictor(model, device, test_loader, criterions, measurements_subset=list(measurement_subset), model_input_info=model_input_info)
        matrix_elements_complex_distance = test_metrics.pop('avg_complex_distance')
        if test_metrics['test_loss'] < best_test_loss:
            best_test_loss = test_metrics['test_loss']
            best_bures_distance = test_metrics['bures_distance']
            model.save(model_save_path)
        # make test_metrics flat
        metrics = {**train_metrics, **test_metrics}
        write_mode = 'w' if epoch == 1 else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=epoch)
        matrix_metrics_dict = {
        '00': matrix_elements_complex_distance[0, 0].item(),
        '01': matrix_elements_complex_distance[0, 1].item(),
        '10': matrix_elements_complex_distance[1, 0].item(),
        '11': matrix_elements_complex_distance[1, 1].item()
        }
        log_metrics_to_file(matrix_metrics_dict, matrix_elements_log_path, write_mode=write_mode, xaxis=epoch)

    plot_metrics_from_file(log_path, title='Loss', save_path=f'./plots/{dir_name}/{model_name}_loss.png')
    return best_test_loss, best_bures_distance


if __name__ == '__main__':
    num_repetitions = 10
    min_num_measurements = 2
    max_num_measurements = 2
    num_qubits = 1
    model_input_info = 'full'
    log_path = f'./logs/1qbit/tomography_corrections_predictor_subset_m2.log'

    batch_size = 64
    train_dataset = MeasurementDataset(root_path='./data/1qbit/train/', return_density_matrix=True, num_qubits=1)
    test_dataset = MeasurementDataset(root_path='./data/1qbit/val/', return_density_matrix=True, num_qubits=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for num_measurements in range(min_num_measurements, max_num_measurements + 1):
        print(f'Running for {num_measurements} measurements')
        dir_name = f'1qbit/tomography_corrections_predictor_m{num_measurements}'
        metrics = {
            'test_loss_avg': 0,
            'test_loss_min': float('inf'),
            'test_loss_max': 0,
            'bures_distance_avg': 0,
            'bures_distance_min': 1.,
            'bures_distance_max': 0,
        }

        used_measurements = []
        num_possible_measurements = len(list(combinations(range(len(Kwiat.basis)**num_qubits), num_measurements)))
        for _ in range(num_repetitions):
            loss, bures_distance = calculate_single_run_metrics(train_loader, test_loader, num_measurements, dir_name, model_input_info, num_qubits=num_qubits, previously_used_measurements=used_measurements)
            metrics['test_loss_avg'] += loss
            metrics['test_loss_min'] = min(metrics['test_loss_min'], loss)
            metrics['test_loss_max'] = max(metrics['test_loss_max'], loss)
            metrics['bures_distance_avg'] += bures_distance
            metrics['bures_distance_min'] = min(metrics['bures_distance_min'], bures_distance)
            metrics['bures_distance_max'] = max(metrics['bures_distance_max'], bures_distance)

        denominator = min(num_repetitions, num_possible_measurements)
        metrics['test_loss_avg'] /= denominator
        metrics['bures_distance_avg'] /= denominator
        write_mode = 'w' if num_measurements == min_num_measurements else 'a'
        # write_mode = 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=num_measurements, xaxis_name='num_measurements')
    plot_metrics_from_file(log_path, title='Metrics', save_path=f'./plots/1qbit/tomography_corrections_predictor_metrics.png', xaxis='num_measurements')
import sys
sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils_measure import Kwiat
from src.datasets import MeasurementDataset
from src.model import LSTMDiscreteMeasurementSelector
from src.torch_utils import train_measurement_predictor, test_discrete_measurement_selector, torch_bures_distance
from src.logging import log_metrics_to_file, plot_metrics_from_file


def main():
    # load data
    batch_size = 128
    test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat.basis]

    # create model
    model_name = 'discrete_lstm_basis_selector_reduced_kwiat_basis_cross_entropy_loss'
    model_save_path = f'./models/{model_name}.pt'
    
    model_params = {
        'num_qubits': 2,
        'possible_basis_matrices': basis_matrices, # 'Kwiat' basis matrices
        'layers': 6,
        'hidden_size': 128,
        'max_num_measurements': 16
    }
    model = LSTMDiscreteMeasurementSelector(**model_params)
    model.load(model_save_path)

    # train & test model
    log_path = f'./logs/{model_name}_measuremnt_dependence.log'
    criterion = nn.MSELoss()
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')
    criterions = {
        'test_loss': criterion,
        'bures_distance': bures_distance
    }
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    test_metrics = test_discrete_measurement_selector(model, device, test_loader, criterions, model_params['max_num_measurements'])
    for i in range(model_params['max_num_measurements']):
        metrics_dict = {metrics_name: test_metrics[metrics_name][f'measurement {i}'] for metrics_name in test_metrics.keys()}
        write_mode = 'w' if i == 0 else 'a'
        log_metrics_to_file(metrics_dict, log_path,  xaxis=i, xaxis_name='num measurements', write_mode=write_mode)
    plot_metrics_from_file(log_path, title='Metrics for measurement disturbance', save_path=f'./plots/{model_name}_measuremnt_dependence.png', xaxis='num measurements')


if __name__ == '__main__':
    main()
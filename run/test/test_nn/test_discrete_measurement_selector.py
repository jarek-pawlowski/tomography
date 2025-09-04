import sys

from src.test_model import test_discrete_measurement_selector
from src.train import train_measurement_predictor
sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.tomography_utils_numpy import Kwiat
from src.datasets import MeasurementDataset
# from src.model import LSTMDiscreteMeasurementSelector
from src.model_optimized import LSTMDiscreteMeasurementSelectorNoMeasurements, LSTMDiscreteMeasurementSelector
from src.criterions import torch_bures_distance
from src.log import log_metrics_to_file, plot_metrics_from_file


def main():
    # load data
    batch_size = 64
    num_qubits = 2
    test_dataset = MeasurementDataset(root_path=f'./data/val/', return_density_matrix=True, num_qubits=num_qubits)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat.basis]

    # create model
    # model_name = 'discrete_lstm_basis_selector_unique_kwiat_basis_cross_entropy_loss_10_noisy_epochs'
    # model_save_path = f'./models/{model_name}.pt'
    model_name = 'discrete_optimized-v2_lstm2_basis_selector_unique_kwiat_basis_cross_entropy_loss_5_noisy_epochs_lr_decreased'
    model_save_path = f'./models/{num_qubits}qbits/{model_name}.pt'
    
    model_params = {
        'num_qubits': num_qubits,
        'possible_basis_matrices': basis_matrices, # 'Kwiat' basis matrices
        'layers': 2,  # 6
        'hidden_size': 256,  # 128
        'max_num_measurements': 4**num_qubits
    }
    # model = LSTMDiscreteMeasurementSelectorNoMeasurements(**model_params)
    model = LSTMDiscreteMeasurementSelector(**model_params)
    model.load(model_save_path)

    # train & test model
    log_path = f'./logs/{num_qubits}qbits/{model_name}_measurement_dependence.log'
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
    plot_metrics_from_file(log_path, title='Metrics for measurement disturbance', save_path=f'./plots/{model_name}_measurement_dependence.png', xaxis='num measurements')


if __name__ == '__main__':
    main()
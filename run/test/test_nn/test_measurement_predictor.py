import sys

from src.test_model import test_measurement_predictor
from src.train import train_measurement_predictor
sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.model import LSTMMeasurementPredictorStackedInput, SequentialMeasurementPredictor, LSTMMeasurementPredictor, LSTMMeasurementPredictorNoSelectionMeausrements
from src.criterions import torch_bures_distance
from src.log import log_metrics_to_file, plot_metrics_from_file


def semidefinite_percentage(predicted_rho: torch.Tensor, target_rho: torch.Tensor) -> float:
    predicted_rho_complex = predicted_rho[..., 0, :, :] + 1j * predicted_rho[..., 1, :, :]
    eigs = torch.amin(torch.linalg.eigvalsh(predicted_rho_complex), dim=-1)
    threshold = -1e-3
    num_positive_eigs = torch.sum(eigs > threshold)
    return num_positive_eigs / eigs.shape[0]


def main():
    # load data
    num_qubits = 2
    batch_size = 128
    test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True, num_qubits=num_qubits)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # create model
    model_name = 'full_lstm_measure_basis'
    # model_name = 'full_lstm_measure_basis_stacked_input-hs1024_no_bases_loss'

    # model_save_path = f'./models/{num_qubits}qbits/{model_name}.pt'
    model_save_path = f'./models/{model_name}.pt'

    model_params = {
        'num_qubits': num_qubits,
        'layers': 6,
        'hidden_size': 128,
        'max_num_measurements': 4**num_qubits
    }
    # model = SequentialMeasurementPredictor(**model_params)
    model = LSTMMeasurementPredictor(**model_params)
    # model = LSTMMeasurementPredictorNoSelectionMeausrements(**model_params)
    # model = LSTMMeasurementPredictorStackedInput(**model_params)
    model.load(model_save_path, map_location=torch.device('cpu'))

    # train & test model
    log_path = f'./logs/{num_qubits}qbits/{model_name}_measurement_dependence.log'
    criterion = nn.MSELoss()
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')
    criterions = {
        'test_loss': criterion,
        'bures_distance': bures_distance,
        'semidefinite_percentage': semidefinite_percentage
    }
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    test_metrics = test_measurement_predictor(model, device, test_loader, criterions, model_params['max_num_measurements'])
    for i in range(model_params['max_num_measurements']):
        metrics_dict = {metrics_name: test_metrics[metrics_name][f'measurement {i}'] for metrics_name in test_metrics.keys()}
        write_mode = 'w' if i == 0 else 'a'
        log_metrics_to_file(metrics_dict, log_path,  xaxis=i, xaxis_name='num measurements', write_mode=write_mode)
    plot_metrics_from_file(log_path, title='Metrics for measurement disturbance', save_path=f'./plots/{model_name}_measurement_dependence_th_1e-3.png', xaxis='num measurements')


if __name__ == '__main__':
    main()
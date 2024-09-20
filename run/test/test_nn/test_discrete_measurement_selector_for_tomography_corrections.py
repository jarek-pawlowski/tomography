import sys

from src.train import train_discrete_measurement_selector
sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.criterions import torch_bures_distance
from src.datasets import MeasurementDataset
from src.model import TomographyCorrectionsLSTMDiscreteMeasurementSelector
from src.test_model import test_discrete_measurement_selector
from src.logging import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat

    
def main():
    # load data
    batch_size = 64
    test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # create model
    model_name = 'tomo_corrections_discrete_lstm_basis_selector_unique_kwiat_basis_cross_entropy_loss'
    model_save_path = f'./models/{model_name}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat.basis]

    model_params = {
        'num_qubits': 2,
        'possible_basis_matrices': basis_matrices, # 'Kwiat' basis matrices
        'layers': 6,
        'hidden_size': 128,
        'max_num_measurements': 16
    }
    model = TomographyCorrectionsLSTMDiscreteMeasurementSelector(**model_params)
    model.load(model_save_path)

    # train & test model
    log_path = f'./logs/{model_name}_measurement_dependence.log'
    
    criterion = nn.MSELoss()
    rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y, reduction='mean'))
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')

    criterions = {
        'test_mse_loss': criterion,
        'test_rmse_loss': rmse_loss,
        'bures_distance': bures_distance
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_metrics = test_discrete_measurement_selector(model, device, test_loader, criterions, model_params['max_num_measurements'], mode='tomo_corrections')
    for i in range(model_params['max_num_measurements']):
        metrics_dict = {metrics_name: test_metrics[metrics_name][f'measurement {i}'] for metrics_name in test_metrics.keys()}
        write_mode = 'w' if i == 0 else 'a'
        log_metrics_to_file(metrics_dict, log_path,  xaxis=i, xaxis_name='num measurements', write_mode=write_mode)
    plot_metrics_from_file(log_path, title='Metrics for measurement disturbance', save_path=f'./plots/{model_name}_measurement_dependence.png', xaxis='num measurements')


if __name__ == '__main__':
    main()
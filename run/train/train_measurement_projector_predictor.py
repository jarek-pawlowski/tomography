import sys

sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset, DerandomizedTestMeasurementDataset
from src.model import LSTMMeasurementProjectorPredictor
from src.criterions import bases_loss, contrastive_bases_loss, contrastive_bases_trace_norm_loss
from src.logging import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat
from src.test_model import test_measurement_projector_predictor
from src.train import train_measurement_projector_predictor


def main():
    # load data
    num_qubits = 2
    batch_size = 64
    train_dataset = MeasurementDataset(root_path='./data/train/', return_density_matrix=True, num_qubits=num_qubits)
    test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True, num_qubits=num_qubits)
    # train_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_train/Xs', mock_label=True)
    # test_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_test/Xs', mock_label=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # create model
    model_name = 'lstm_measurement_projector_predictor'
    model_save_path = f'./models/{num_qubits}qbits/{model_name}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat.basis]

    def kwiat_basis_loss_fn(predicted_bases: torch.Tensor) -> torch.Tensor:
        return bases_loss(predicted_bases, torch.stack(basis_matrices), reduction='mean')
    
    model_params = {
        'num_qubits': num_qubits,
        'hidden_size': 128,
        'max_num_measurements': 4**num_qubits,
    }

    model = LSTMMeasurementProjectorPredictor(**model_params)
    
    # Load model if want to continue training
    # model.load(model_save_path)

    # train & test model
    log_path = f'./logs/{num_qubits}qbits/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    num_epochs = 40
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    criterions = {
        'test_loss': criterion
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_test_loss = float('inf')
    last_measurement_idx = 4**num_qubits - 1
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_measurement_projector_predictor(model, device, train_loader, optimizer, epoch, criterion=criterion, log_interval=10, increase_loss_weights_with_measurement=False) #, bases_loss_fn=contrastive_bases_trace_norm_loss, bases_loss_weight=0.001, contrastive_loss_start_epoch=4)
        test_metrics = test_measurement_projector_predictor(model, device, test_loader, criterions, model_params['max_num_measurements'])
        if test_metrics['test_loss'][f'measurement {last_measurement_idx}'] < best_test_loss:
            best_test_loss = test_metrics['test_loss'][f'measurement {last_measurement_idx}']
            model.save(model_save_path)
        # make test_metrics flat
        test_metrics = {f'{name}_{subname}': value for name, metrics in test_metrics.items() for subname, value in metrics.items()}
        metrics = {**train_metrics, **test_metrics}
        write_mode = 'w' if epoch == 1 else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=epoch)
    plot_metrics_from_file(log_path, title='Loss', save_path=f'./plots/{num_qubits}qbits/{model_name}_loss.png')


if __name__ == '__main__':
    main()
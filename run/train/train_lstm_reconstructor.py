import sys

sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset, DerandomizedTestMeasurementDataset
from src.model import LSTMDensityMatrixReconstructor
from src.train import train_lstm_reconstructor
from src.test_model import test_lstm_reconstructor
from src.log import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat


def list_to_str(l):
    return '_'.join([str(x) for x in l])
    
def main():
    # load data
    batch_size = 64
    num_qubits = 2
    train_dataset = MeasurementDataset(root_path='./data/train/', return_density_matrix=True, num_qubits=num_qubits)
    test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True, num_qubits=num_qubits)
    # train_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_train/Xs', mock_label=True)
    # test_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_test/Xs', mock_label=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    measurements_order = torch.randperm(4**num_qubits, device=device)

    model_name = 'lstm_reconstructor'
    model_name = f'{model_name}_m{list_to_str(measurements_order.cpu().tolist())}'

    model_save_path = f'./models/{num_qubits}qbits/{model_name}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    model_params = {
        'input_dim': 2 * (num_qubits*4) + 1,
        'num_qubits': num_qubits,
        'layers': 2,
        'hidden_size': 256,
        'bias': True
    }

    model = LSTMDensityMatrixReconstructor(**model_params)

    # train & test model
    log_path = f'./logs/{num_qubits}qbits/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    num_epochs = 40
    reconstructor_optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    criterions = {
        'test_loss': criterion
    }


    best_test_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_lstm_reconstructor(model, device, train_loader, reconstructor_optimizer, epoch, measurements_order, criterion=criterion, log_interval=10)
        test_metrics = test_lstm_reconstructor(model, device, test_loader, criterions, measurements_order)
        if test_metrics['test_loss'][f'measurement {4**num_qubits - 1}'] < best_test_loss:
            best_test_loss = test_metrics['test_loss'][f'measurement {4**num_qubits - 1}']
            model.save(model_save_path)
        # make test_metrics flat
        test_metrics = {f'{name}_{subname}': value for name, metrics in test_metrics.items() for subname, value in metrics.items()}
        metrics = {**train_metrics, **test_metrics}
        write_mode = 'w' if epoch == 1 else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=epoch)
    plot_metrics_from_file(log_path, title='Loss', save_path=f'./plots/{num_qubits}qbits/{model_name}_loss.png')


if __name__ == '__main__':
    main()
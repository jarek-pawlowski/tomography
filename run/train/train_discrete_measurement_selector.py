import sys

sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset, DerandomizedTestMeasurementDataset
from src.model import LSTMDiscreteMeasurementSelector, LSTMDiscreteMeasurementSelectorOptimized
from src.train import train_discrete_measurement_selector, train_optimized_discrete_measurement_selector
from src.test_model import test_discrete_measurement_selector
from src.log import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat

    
def main():
    # load data
    batch_size = 64
    num_qubits = 2
    # train_dataset = MeasurementDataset(root_path='./data/3qbits/train/', return_density_matrix=True, num_qubits=num_qubits)
    # test_dataset = MeasurementDataset(root_path='./data/3qbits/val/', return_density_matrix=True, num_qubits=num_qubits)
    train_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_train/Xs', mock_label=True)
    test_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_test/Xs', mock_label=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # create model
    pretrained_model_name = 'Xs_discrete_lstm_basis_selector_unique_kwiat_basis_cross_entropy_loss_10_noisy_epochs'
    pretrained_model_save_path = f'./models/{num_qubits}qbits/{pretrained_model_name}.pt'

    model_name = 'pretrained_Xs_discrete_lstm_basis_selector_unique_kwiat_basis_cross_entropy_loss_10_noisy_epochs_ordered_equal-lr'
    model_save_path = f'./models/{num_qubits}qbits/{model_name}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat.basis]

    model_params = {
        'num_qubits': num_qubits,
        'possible_basis_matrices': basis_matrices, # 'Kwiat' basis matrices
        'layers': 6,
        'hidden_size': 128,
        'max_num_measurements': 4**num_qubits
    }
    # model = LSTMDiscreteMeasurementSelectorOptimized(**model_params)
    model = LSTMDiscreteMeasurementSelector(**model_params)
    model.load(pretrained_model_save_path)

    # train & test model
    log_path = f'./logs/{num_qubits}qbits/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    num_epochs = 40
    reconstructor_optimizer = optim.Adam(model.matrix_reconstructor.parameters(), lr=0.01)
    selector_optimizer = optim.Adam(list(model.measurement_selector.parameters()) + list(model.projectors.parameters()), lr=0.01)
    criterion = nn.MSELoss()
    selector_criterion = nn.CrossEntropyLoss()
    criterions = {
        'test_loss': criterion
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_test_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        # train_metrics = train_optimized_discrete_measurement_selector(model, device, train_loader, reconstructor_optimizer, selector_optimizer, epoch, reconstructor_criterion=criterion, selector_criterion=selector_criterion, log_interval=10, num_reconstructor_repeats=1, num_selector_repeats=1, num_noisy_epochs=0)
        train_metrics = train_discrete_measurement_selector(model, device, train_loader, reconstructor_optimizer, selector_optimizer, epoch, reconstructor_criterion=criterion, selector_criterion=selector_criterion, log_interval=10, num_reconstructor_repeats=1, num_selector_repeats=1, num_noisy_epochs=0, selector_train_mode='ordered')
        test_metrics = test_discrete_measurement_selector(model, device, test_loader, criterions, model_params['max_num_measurements'])
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
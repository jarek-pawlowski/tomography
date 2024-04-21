import sys
sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.model_shadow import LSTMMeasurementSelector
from src.torch_utils import train_measurement_predictor, test_measurement_predictor, bases_loss
from src.logging import log_metrics_to_file, plot_metrics_from_file
from src.utils_measure import Pauli, Pauli_c, Pauli_vector

    
def main():
    # load data
    batch_size = 128
    train_dataset = MeasurementDataset(root_path='./small_data/train/', return_density_matrix=True)
    test_dataset = MeasurementDataset(root_path='./small_data/train/', return_density_matrix=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create model
    model_name = 'full_lstm_basis_selector_v3_Pauli_basis_loss'
    model_save_path = f'./models/{model_name}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli.basis]
    basis_matrices_c = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli_c.basis]
    basis_reconstruction = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli_vector]

    def basis_loss_fn(predicted_bases: torch.Tensor) -> torch.Tensor:
        return bases_loss(predicted_bases, torch.stack(basis_matrices), reduction='mean')
    
    model_params = {
        'num_qubits': 2,
        'basis_matrices': basis_matrices, # 'Pauli' basis matrices
        'basis_matrices_c': basis_matrices_c, # 'Pauli complementary' basis matrices
        'basis_reconstruction': basis_reconstruction,
        'layers': 6,
        'hidden_size': 128,
        'max_num_snapshots': 100,
        'device': device
    }
    model = LSTMMeasurementSelector(**model_params)

    # train & test model
    log_path = f'./logs/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    num_epochs = 10
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    criterions = {
        'test_loss': criterion
    }

    best_test_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_measurement_predictor(model, device, train_loader, optimizer, epoch, criterion=criterion, log_interval=10, bases_loss_fn=basis_loss_fn)
        test_metrics = test_measurement_predictor(model, device, test_loader, criterions, 1)
        # if test_metrics['test_loss']['measurement 15'] < best_test_loss:
        #     best_test_loss = test_metrics['test_loss']['measurement 15']
        #     model.save(model_save_path)
        # make test_metrics flat
        test_metrics = {f'{name}_{subname}': value for name, metrics in test_metrics.items() for subname, value in metrics.items()}
        metrics = {**train_metrics, **test_metrics}
        write_mode = 'w' if epoch == 1 else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=epoch)
    plot_metrics_from_file(log_path, title='Loss', save_path=f'./plots/{model_name}_loss.png')


if __name__ == '__main__':
    main()
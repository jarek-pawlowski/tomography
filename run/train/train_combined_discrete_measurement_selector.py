import sys

from src.criterions import torch_bures_distance

sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset, DerandomizedTestMeasurementDataset
from src.model_optimized import CombinedLSTMDiscreteMeasurementSelector, CombinedLSTMMeasurementPredictor
from src.train import train_combined_lstm
from src.test_model import test_combined_lstm
from src.log import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat

    
def main():
    # load data
    batch_size = 512
    num_qubits = 4
    train_dataset = MeasurementDataset(root_path='./data/4qbits/train/', return_density_matrix=True, num_qubits=num_qubits)
    test_dataset = MeasurementDataset(root_path='./data/4qbits/val/', return_density_matrix=True, num_qubits=num_qubits)
    # train_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_train/Xs', mock_label=True)
    # test_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_test/Xs', mock_label=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model_name = 'combined_lstm2_discrete_unique_measurement_selector_no_measurements_measure_memory'
    # model_name = 'combined_lstm2_semi_rand_measurement_predictor_measure_memory'

    model_save_path = f'./models/{num_qubits}qbits/{model_name}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model_params = {
        'num_qubits': num_qubits,
        'hidden_size': 256,
        'max_num_measurements': 4**num_qubits,
        'selection_measurements': False,
        'num_layers': 2,
        # 'temperature': 1.
    }

    # model = CombinedLSTMMeasurementPredictor(**model_params)
    model = CombinedLSTMDiscreteMeasurementSelector(**model_params)

    # train & test model
    log_path = f'./logs/{num_qubits}qbits/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    num_epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
    
    criterion = nn.MSELoss()
    criterions = {
        'test_loss_m': criterion
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # best_test_loss = float('inf')
    # for epoch in range(1, num_epochs + 1):
    #     train_metrics = train_combined_lstm(model, device, train_loader, optimizer, epoch, log_interval=10)
    #     with torch.no_grad():
    #         test_metrics = test_combined_lstm(model, device, test_loader, criterions)
        
    #     if test_metrics['test_loss_m'][f'measurement {4**num_qubits - 1}'] < best_test_loss:
    #         best_test_loss = test_metrics['test_loss_m'][f'measurement {4**num_qubits - 1}']
    #         model.save(model_save_path)

    #     test_loss_total = test_metrics.pop('test_loss')
    #     # make test_metrics flat
    #     test_metrics = {f'{name}_{subname}': value for name, metrics in test_metrics.items() for subname, value in metrics.items()}
        
    #     metrics = {
    #         **train_metrics,
    #         'test_loss_sum': test_loss_total,
    #         **test_metrics
    #     }
    #     write_mode = 'w' if epoch == 1 else 'a'
    #     log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=epoch)

    #     scheduler.step()
    
    # plot_metrics_from_file(log_path, title='Loss', save_path=f'./plots/{num_qubits}qbits/{model_name}_loss.png')

    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')

    final_criterions = {
        'test_loss_m': criterion,
        'bures_distance': bures_distance
    }
    final_log_path = f'./logs/{num_qubits}qbits/{model_name}_measurement_dependence.log'

    model.load(model_save_path)
    with torch.no_grad():
        best_metrics = test_combined_lstm(model, device, test_loader, final_criterions)

    for i in range(model_params['max_num_measurements']):
        write_mode = 'w' if i == 0 else 'a'
        metrics_i = {
            'test_loss': best_metrics['test_loss_m'][f'measurement {i}'],
            'bures_distance': best_metrics['bures_distance'][f'measurement {i}']
        }
        log_metrics_to_file(metrics_i, final_log_path, write_mode=write_mode, xaxis=i + 1, xaxis_name='num_measurements')




if __name__ == '__main__':
    main()
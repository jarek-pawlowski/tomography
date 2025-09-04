import sys

sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.multiprocessing import Queue, Process, set_start_method

from src.criterions import torch_bures_distance
from src.datasets import MeasurementDataset, DerandomizedTestMeasurementDataset
from src.model import LSTMDensityMatrixReconstructor
from src.train import train_lstm_reconstructor
from src.test_model import test_lstm_reconstructor
from src.log import log_metrics_to_file, plot_metrics_from_file
from src.tomography_utils_numpy import Kwiat


def list_to_str(l):
    return '_'.join([str(x) for x in l])


def generate_random_measurements_subsets(num_repetitions: int, num_qubits: int):
    used_measurements = []
    for _ in range(num_repetitions):
        while (measurement_subset := torch.randperm(4**num_qubits).tolist()) in used_measurements:
            pass
        used_measurements.append(measurement_subset)
    return used_measurements


def calculate_single_run_metrics(
    result_queue: Queue,
    mid: int,
    dir_name: str,
    num_qubits: int,
    measurements_order: list
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    train_dataset = MeasurementDataset(root_path='./data/4qbits/train/', return_density_matrix=True, num_qubits=num_qubits)
    test_dataset = MeasurementDataset(root_path='./data/4qbits/val/', return_density_matrix=True, num_qubits=num_qubits)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model_name = 'lstm_reconstructor'
    model_name = f'{model_name}_mid{mid}'

    sys.stdout = open(f"./logs/debug/{model_name}.log", 'w')

    model_save_path = f'./models/{dir_name}/{model_name}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    model_params = {
        'input_dim': 2 * (num_qubits*4) + 1,
        'num_qubits': num_qubits,
        'layers': 2,
        'hidden_size': 1024,
        'bias': True
    }

    model = LSTMDensityMatrixReconstructor(**model_params)

    # train & test model
    log_path = f'./logs/{dir_name}/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    num_epochs = 30
    reconstructor_optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')
    criterions = {
        'test_loss': criterion,
    }

    measurements_order_t = torch.tensor(measurements_order, device=device)

    best_test_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_metrics = train_lstm_reconstructor(model, device, train_loader, reconstructor_optimizer, epoch, measurements_order_t, criterion=criterion, log_interval=10, std_out=sys.stdout)
        test_metrics = test_lstm_reconstructor(model, device, test_loader, criterions, measurements_order_t, std_out=sys.stdout)
        if test_metrics['test_loss'][f'measurement {4**num_qubits - 1}'] < best_test_loss:
            best_test_loss = test_metrics['test_loss'][f'measurement {4**num_qubits - 1}']
            model.save(model_save_path)
        # make test_metrics flat
        test_metrics = {f'{name}_{subname}': value for name, metrics in test_metrics.items() for subname, value in metrics.items()}
        metrics = {**train_metrics, **test_metrics}
        write_mode = 'w' if epoch == 1 else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=epoch)
    plot_metrics_from_file(log_path, title='Loss', save_path=f'./plots/{num_qubits}qbits/{model_name}_loss.png')
    

    # Final evaluation
    final_criterions = {
        'test_loss': criterion,
        'bures_distance': bures_distance
    }

    model.load(model_save_path)
    best_metrics = test_lstm_reconstructor(model, device, test_loader, final_criterions, measurements_order_t, std_out=sys.stdout)

    test_loss_t = [
        best_metrics['test_loss'][f'measurement {i}'] for i in range(4**num_qubits)
    ]
    bures_distance_t = [
        best_metrics['bures_distance'][f'measurement {i}'] for i in range(4**num_qubits)
    ]

    # result_metrics = result_queue.get()
    # result_metrics['test_loss_avg'] += test_loss_t
    # result_metrics['bures_distance_avg'] += bures_distance_t

    result_metrics = {
        'test_loss_avg': test_loss_t,
        'bures_distance_avg': bures_distance_t
    }
    result_queue.put(result_metrics)


if __name__ == '__main__':
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    num_repetitions = 3
    num_qubits = 4
    num_measurements = 4 ** num_qubits
    log_path = f'./logs/{num_qubits}qbits/lstm_reconstructor.log'

    dir_name = f'{num_qubits}qbits/lstm_reconstructor'
    # Parent-side accumulators
    queue = Queue()

    # metrics = {
    #     'test_loss_avg': torch.zeros(num_measurements),
    #     'bures_distance_avg': torch.zeros(num_measurements)
    # }

    # queue.put(metrics)

    measurement_sets = generate_random_measurements_subsets(num_repetitions, num_qubits)
    processes = [Process(target=calculate_single_run_metrics, args=(queue, i, dir_name, num_qubits, measurement_sets[i])) for i in range(num_repetitions)]

    for p in processes:
        p.start()
        print(f'Process {p.pid} started')

    for p in processes:
        p.join()

    # Collect results
    # metrics = queue.get()

    # Collect results
    metrics = {
        'test_loss_avg': torch.zeros(num_measurements),
        'bures_distance_avg': torch.zeros(num_measurements)
    }
    for i in range(num_repetitions):
        result = queue.get()
        metrics['test_loss_avg'] += torch.tensor(result['test_loss_avg'])
        metrics['bures_distance_avg'] += torch.tensor(result['bures_distance_avg'])

    # Aggregate and log
    test_loss_avg = metrics['test_loss_avg'] / num_repetitions
    bures_distance_avg = metrics['bures_distance_avg'] / num_repetitions

    for i in range(num_measurements):
        write_mode = 'w' if i == 0 else 'a'
        metrics_i = {
            'test_loss_avg': test_loss_avg[i].item(),
            'bures_distance_avg': bures_distance_avg[i].item(),
        }
        log_metrics_to_file(metrics_i, log_path, write_mode=write_mode, xaxis=i + 1, xaxis_name='num_measurements')

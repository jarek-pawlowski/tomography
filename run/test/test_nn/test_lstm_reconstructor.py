import sys

sys.path.append('./')
import os

import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.multiprocessing import Queue, Process, set_start_method

from src.criterions import torch_bures_distance
from src.datasets import MeasurementDataset, DerandomizedTestMeasurementDataset
from src.model import LSTMDensityMatrixReconstructor
from src.model_optimized import LSTMReconstructor
from src.train import train_lstm_reconstructor, train_lstm_reconstructor_optimized
from src.test_model import test_lstm_reconstructor, test_lstm_reconstructor_optimized
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

    batch_size = 512
    test_dataset = MeasurementDataset(root_path='./data/4qbits/val/', return_density_matrix=True, num_qubits=num_qubits)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model_name = 'lstm2_reconstructor_optimized'
    model_name = f'{model_name}_mid{mid}'

    sys.stdout = open(f"./logs/debug/{model_name}.log", 'w')

    model_save_path = f'./models/{dir_name}/{model_name}.pt'
    
    model_params = {
        # 'input_dim': 2 * (num_qubits*4) + 1,
        'num_qubits': num_qubits,
        # 'layers': 2,
        'hidden_size': 256,
        'max_num_measurements': 4**num_qubits,
        'num_layers': 2,
        # 'bias': True
    }

    # model = LSTMDensityMatrixReconstructor(**model_params)
    model = LSTMReconstructor(**model_params)

    # train & test model
    log_path = f'./logs/{dir_name}/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    criterion = nn.MSELoss()
    bures_distance = lambda x, y: torch_bures_distance(x, y, reduction='mean')


    measurements_order_t = torch.tensor(measurements_order, device=device)


    # Final evaluation
    final_criterions = {
        'test_loss_m': criterion,
        'bures_distance': bures_distance
    }

    model.load(model_save_path)
    with torch.no_grad():
        best_metrics = test_lstm_reconstructor_optimized(model, device, test_loader, measurements_order_t, criterions=final_criterions, std_out=sys.stdout)
    test_loss_total = best_metrics.pop('test_loss')

    test_loss_t = [
        best_metrics['test_loss_m'][f'measurement {i}'] for i in range(4**num_qubits)
    ]
    bures_distance_t = [
        best_metrics['bures_distance'][f'measurement {i}'] for i in range(4**num_qubits)
    ]

    # result_metrics = result_queue.get()
    # result_metrics['test_loss_avg'] += test_loss_t
    # result_metrics['bures_distance_avg'] += bures_distance_t

    result_metrics = {
        'test_loss_avg': test_loss_t,
        'bures_distance_avg': bures_distance_t,
        'test_loss_sum': test_loss_total
    }
    result_queue.put(result_metrics)


if __name__ == '__main__':
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    model_ids = [1, 2]
    num_qubits = 4
    num_measurements = 4 ** num_qubits
    log_path = f'./logs/{num_qubits}qbits/lstm2_reconstructor_optimized.log'

    dir_name = f'{num_qubits}qbits/lstm2_reconstructor_optimized'
    
    # Parent-side accumulators
    queue = Queue()

    # metrics = {
    #     'test_loss_avg': torch.zeros(num_measurements),
    #     'bures_distance_avg': torch.zeros(num_measurements)
    # }

    # queue.put(metrics)

    measurements_save_path = f'./models/{dir_name}/measurement_sets.pkl'
    os.makedirs(os.path.dirname(measurements_save_path), exist_ok=True)
    
    # measurement_sets = generate_random_measurements_subsets(num_repetitions, num_qubits)
    measurement_sets = pickle.load(open(measurements_save_path, 'rb'))

    processes = [Process(target=calculate_single_run_metrics, args=(queue, i, dir_name, num_qubits, measurement_sets[i])) for i in model_ids]

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
    for i in model_ids:
        result = queue.get()
        metrics['test_loss_avg'] += torch.tensor(result['test_loss_avg'])
        metrics['bures_distance_avg'] += torch.tensor(result['bures_distance_avg'])

    # Aggregate and log
    test_loss_avg = metrics['test_loss_avg'] / len(model_ids)
    bures_distance_avg = metrics['bures_distance_avg'] / len(model_ids)

    for i in range(num_measurements):
        write_mode = 'w' if i == 0 else 'a'
        metrics_i = {
            'test_loss_avg': test_loss_avg[i].item(),
            'bures_distance_avg': bures_distance_avg[i].item(),
        }
        log_metrics_to_file(metrics_i, log_path, write_mode=write_mode, xaxis=i + 1, xaxis_name='num_measurements')

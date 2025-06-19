import sys

from tqdm import tqdm
import numpy as np

from src.test_model import test
sys.path.append('./')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.model import Regressor, Classifier
from src.criterions import regressor_accuracy, regressor_precision, regressor_recall, regressor_balanced_accuracy
from src.log import log_metrics_to_file, plot_metrics_from_file

measurement_subsets_4 = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
measurement_subsets_12 = [[i for i in range(16) if i not in subset] for subset in measurement_subsets_4]
measurement_subsets = measurement_subsets_4 + measurement_subsets_12
# measurement_subset = None

def parse_measurement_subset(measurement_subset):
    str_list = '_'.join([str(x) for x in measurement_subset])
    return f'm{str_list}'

batch_size = 512

for measurement_subset in measurement_subsets:
    print(f'Running test for measurement subset: {measurement_subset}')
    test_dataset = MeasurementDataset(root_path='./data/val/', measurement_subset=measurement_subset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # model_name = 'regressor'
    model_name = f'regressor_{parse_measurement_subset(measurement_subset)}'
    model_path = f'./models/{model_name}.pt'
    model_params = {
        'input_dim': len(measurement_subset),
        'output_dim': 1,
        'layers': 2,
        'hidden_size': 128,
        'input_dropout': 0.
    }
    model = Regressor(**model_params)
    # model = Classifier(**model_params)
    model.load(model_path, map_location='cpu')

    th_range = (0.01, 0.1)
    param_space = np.linspace(*th_range, num=10)

    for i, th in enumerate(param_space):
        print(f'Threshold: {th}')
        rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y))
        mse_loss = nn.MSELoss()
        accuracy = lambda x, y: regressor_accuracy(x, y, input_threshold=th, target_threshold=1.e-6)
        balanced_accuracy = lambda x, y: regressor_balanced_accuracy(x, y, input_threshold=th, target_threshold=1.e-6)
        precision = lambda x, y: regressor_precision(x, y, input_threshold=th, target_threshold=1.e-6)
        recall = lambda x, y: regressor_recall(x, y, input_threshold=th, target_threshold=1.e-6)
        criterions = {
            'test_rmse_loss': rmse_loss,
            'test_mse_loss': mse_loss,
            'test_accuracy': accuracy,
            'test_balanced_accuracy': balanced_accuracy,
            'test_precision': precision,
            'test_recall': recall
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_metrics = test(model, device, test_loader, criterions)

        if i == 0:
            log_metrics_to_file(test_metrics, f'./logs/{model_name}_test_1e-6.log', write_mode='w', xaxis=th, xaxis_name='threshold')
        else:
            log_metrics_to_file(test_metrics, f'./logs/{model_name}_test_1e-6.log', write_mode='a', xaxis=th, xaxis_name='threshold')
    print('\n')
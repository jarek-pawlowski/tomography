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
from src.criterions import regressor_accuracy, regressor_precision, regressor_recall, regressor_balanced_accuracy, bins_criterion
from src.logging import log_metrics_to_file, plot_metrics_from_file

batch_size = 512

test_dataset = MeasurementDataset(root_path='./data/val/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model_name = 'regressor'
model_path = f'./models/{model_name}.pt'
model_params = {
    'input_dim': 16,
    'output_dim': 1,
    'layers': 2,
    'hidden_size': 128,
    'input_dropout': 0.
}
model = Regressor(**model_params)
# model = Classifier(**model_params)
model.load(model_path, map_location='cpu')

concurrence_boundaries = np.linspace(0., 1., 100)
concurrence_boundaries[0] = 1.e-6

rmse_loss = lambda x, y: bins_criterion(
    x, y, concurrence_boundaries, lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y))
)
mse_loss = lambda x, y: bins_criterion(
    x, y, concurrence_boundaries, nn.MSELoss()
)
accuracy = lambda x, y: bins_criterion(
    x, y, concurrence_boundaries, regressor_accuracy, {'input_threshold': 0.03, 'target_threshold': 1.e-6}
)
balanced_accuracy = lambda x, y: bins_criterion(
    x, y, concurrence_boundaries, regressor_balanced_accuracy, {'input_threshold': 0.03, 'target_threshold': 1.e-6}
)
precision = lambda x, y: bins_criterion(
    x, y, concurrence_boundaries, regressor_precision, {'input_threshold': 0.03, 'target_threshold': 1.e-6}
)
recall = lambda x, y: bins_criterion(
    x, y, concurrence_boundaries, regressor_recall, {'input_threshold': 0.03, 'target_threshold': 1.e-6}
)

criterions = {
    'test_rmse_loss': rmse_loss,
    'test_mse_loss': mse_loss,
    'test_accuracy': accuracy,
    'test_balanced_accuracy': balanced_accuracy,
    'test_precision': precision,
    'test_recall': recall
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_metrics = test(model, device, test_loader, binlike_criterions=criterions, num_bins=len(concurrence_boundaries))

for i in range(len(concurrence_boundaries) - 1):
    metrics_i = {key: value[i+1] for key, value in test_metrics.items()}
    concurrence = (concurrence_boundaries[i] + concurrence_boundaries[i+1]) / 2
    if i == 0:
        log_metrics_to_file(metrics_i, f'./logs/{model_name}_test_conc.log', write_mode='w', xaxis=concurrence, xaxis_name='concurrence')
    else:
        log_metrics_to_file(metrics_i, f'./logs/{model_name}_test_conc.log', write_mode='a', xaxis=concurrence, xaxis_name='concurrence')

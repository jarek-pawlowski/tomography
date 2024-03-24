import sys
sys.path.append('./')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset, VectorDensityMatrixDataset
from src.model import Regressor, Classifier
from src.torch_utils import test_varying_input, regressor_accuracy
from src.logging import log_metrics_to_file, plot_metrics_from_file, plot_metrics_from_files

batch_size = 512
test_dataset = MeasurementDataset(root_path='./data/val/')
# test_dataset = VectorDensityMatrixDataset(root_path='./data/val/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

results_path_prefix = './logs/regressor_varying_measurements/regressor_test_varying_measurements_clipped_extended_'
results_path = '{}{}.log'
plot_path = './plots/regressor_varying_measurements/regressor_test_varying_measurements_clipped_extended_{}.png'

model_path = './models/regressor.pt'
model_params = {
    'input_dim': 16,
    'output_dim': 1,
    'layers': 2,
    'hidden_size': 128,
    'input_dropout': 0.0
}
model = Regressor(**model_params)
# model = Classifier(**model_params)
model.load(model_path)

rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y, reduction='none'))
relative_rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y, reduction='none')) / (y + 1e-5)
clipped_rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x.clamp(0, 1), y, reduction='none'))
mse_loss = nn.MSELoss(reduction='none')
bce_loss = nn.BCELoss(reduction='none')
accuracy = lambda x, y: regressor_accuracy(x, y, input_threshold=0.5, target_threshold=0.5, reduction='none')
criterions = {
    'test_rmse_loss': rmse_loss,
    'test_mse_loss': mse_loss,
    'test_relative_rmse_loss': relative_rmse_loss,
    'test_clipped_rmse_loss': clipped_rmse_loss,
    # 'test_bce_loss': bce_loss,
    'test_accuracy': accuracy
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(0, model_params['input_dim']):
    print('Measurement', i)
    test_metrics = test_varying_input(model, device, test_loader, criterions, varying_input_idx=[i], max_variance=1., step=0.05)
    for variance, metrics in test_metrics.items():
        write_mode = 'w' if variance == 0 else 'a'
        log_metrics_to_file(metrics, results_path.format(results_path_prefix,  f'{i}'), write_mode=write_mode, xaxis=variance, xaxis_name='variance')
    plot_metrics_from_file(results_path.format(results_path_prefix,  f'{i}'), title=f'Metrics for measurement {i}', save_path=plot_path.format(f'{i}'), xaxis='variance')

plot_metrics_from_files(f'{results_path_prefix}', (0, model_params['input_dim']), title=f'RMSE loss for varying measurements', save_path=plot_path.format(f'{i}_rmse'), xaxis='variance', specified_metric='test_rmse_loss')
plot_metrics_from_files(f'{results_path_prefix}', (0, model_params['input_dim']), title=f'Clipped RMSE loss for varying measurements', save_path=plot_path.format(f'{i}_rmse_clipped'), xaxis='variance', specified_metric='test_clipped_rmse_loss')
plot_metrics_from_files(f'{results_path_prefix}', (0, model_params['input_dim']), title=f'Relative RMSE loss for varying measurements', save_path=plot_path.format(f'{i}_rmse_relative'), xaxis='variance', specified_metric='test_relative_rmse_loss')
plot_metrics_from_files(f'{results_path_prefix}', (0, model_params['input_dim']), title=f'Accuracy for varying measurements', save_path=plot_path.format(f'{i}_acc'), xaxis='variance', specified_metric='test_accuracy')

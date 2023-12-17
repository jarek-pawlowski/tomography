import sys
sys.path.append('./')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.torch_measure import test_concurrence_measurement_noise
from src.torch_utils import test_varying_input, regressor_accuracy
from src.logging import log_metrics_to_file, plot_metrics_from_file, plot_metrics_from_files

batch_size = 512
test_dataset = MeasurementDataset(root_path='./data/val/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

results_path_prefix = './logs/concurrence_varying_measurements/concurrence_test_varying_measurement_clipped_'
results_path = '{}{}.log'
plot_path = './plots/concurrence_varying_measurements/concurrence_test_varying_measurement_clipped_{}.png'

num_measurements = 16

rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y, reduction='none'))
mse_loss = nn.MSELoss(reduction='none')
accuracy = lambda x, y: regressor_accuracy(x, y, input_threshold=0.05, target_threshold=0.001, reduction='none')
criterions = {
    'test_rmse_loss': rmse_loss,
    'test_mse_loss': mse_loss,
    'test_accuracy': accuracy
}

for i in range(0, num_measurements):
    print('Measurement input', i)
    test_metrics = test_concurrence_measurement_noise(test_loader, criterions, varying_input_idx=[i], max_variance=1., step=0.05)
    for variance, metrics in test_metrics.items():
        write_mode = 'w' if variance == 0 else 'a'
        log_metrics_to_file(metrics, results_path.format(results_path_prefix, i), write_mode=write_mode, xaxis=variance, xaxis_name='variance')
    plot_metrics_from_file(results_path.format(results_path_prefix, i), title=f'Metrics for measurement {i}', save_path=plot_path.format(i), xaxis='variance')

plot_metrics_from_files(results_path_prefix, (0, num_measurements), title='RMSE loss for varying measurements', save_path=plot_path.format('rmse'), xaxis='variance', specified_metric='test_rmse_loss')
plot_metrics_from_files(results_path_prefix, (0, num_measurements), title='Accuracy for varying measurements', save_path=plot_path.format('acc'), xaxis='variance', specified_metric='test_accuracy')

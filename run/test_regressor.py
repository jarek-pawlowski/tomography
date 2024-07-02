import sys
sys.path.append('./')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.model import Regressor
from src.torch_utils import test, regressor_accuracy
from src.logging import log_metrics_to_file, plot_metrics_from_file

batch_size = 512
test_dataset = MeasurementDataset(root_path='./data/val/', measurement_subset=[2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model_name = 'regressor_m2_3_6_7_8_9_10_11_12_13_14_15'
model_path = f'./models/{model_name}.pt'
model_params = {
    'input_dim': 12,
    'output_dim': 1,
    'layers': 2,
    'hidden_size': 128,
    'input_dropout': 0.
}
model = Regressor(**model_params)
model.load(model_path)

rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y))
mse_loss = nn.MSELoss()
accuracy = lambda x, y: regressor_accuracy(x, y, input_threshold=0.05, target_threshold=0.001)
criterions = {
    'test_rmse_loss': rmse_loss,
    'test_mse_loss': mse_loss,
    'test_accuracy': accuracy
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_metrics = test(model, device, test_loader, criterions)
log_metrics_to_file(test_metrics, f'./logs/{model_name}_test.log', write_mode='w')

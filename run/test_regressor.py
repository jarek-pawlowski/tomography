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
test_dataset = MeasurementDataset(root_path='./data/val/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model_path = './models/regressor.pt'
model_params = {
    'input_dim': 16,
    'output_dim': 1,
    'layers': 2,
    'hidden_size': 128,
    'input_dropout': 0.0
}
model = Regressor(**model_params)
model.load(model_path)

rmse_loss = lambda x, y: torch.sqrt(torch.functional.F.mse_loss(x, y))
mse_loss = nn.MSELoss()
accuracy = lambda x, y: regressor_accuracy(x, y, input_threshold=0.01, target_threshold=0.01)
criterions = {
    'test_rmse_loss': rmse_loss,
    'test_mse_loss': mse_loss,
    'test_accuracy': accuracy
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_metrics = test(model, device, test_loader, criterions)
log_metrics_to_file(test_metrics, './logs/regressor_test.log', write_mode='w')

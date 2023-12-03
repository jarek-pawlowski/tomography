import sys
sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset
from src.model import Regressor
from src.torch_utils import train, test
from src.logging import log_metrics_to_file, plot_metrics_from_file


def main():
    # load data
    batch_size = 512
    train_dataset = MeasurementDataset(root_path='./data/train/')
    test_dataset = MeasurementDataset(root_path='./data/val/')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # create model
    model_save_path = './models/regressor_10_fcl.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    model_params = {
        'input_dim': 16,
        'output_dim': 1,
        'layers': 10,
        'hidden_size': 128,
        'input_dropout': 0.
    }
    model = Regressor(**model_params)

    # train & test model
    log_path = './logs/regressor_10_fcl.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    num_epochs = 40
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    criterions = {
        'test_loss': criterion
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_test_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_metrics = train(model, device, train_loader, optimizer, epoch, criterion=criterion, log_interval=10)
        test_metrics = test(model, device, test_loader, criterions)
        if test_metrics['test_loss'] < best_test_loss:
            best_test_loss = test_metrics['test_loss']
            model.save(model_save_path)
        metrics = {**train_metrics, **test_metrics}
        write_mode = 'w' if epoch == 1 else 'a'
        log_metrics_to_file(metrics, log_path, write_mode=write_mode, xaxis=epoch)
    plot_metrics_from_file(log_path, title='Loss', save_path='./plots/regressor_10_fcl_loss.png')


if __name__ == '__main__':
    main()
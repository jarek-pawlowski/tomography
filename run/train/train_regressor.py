import sys

from src.train import train
sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset, VectorDensityMatrixDataset, FilteredDataset
from src.model import Regressor, Classifier
from src.test_model import test
from src.log import log_metrics_to_file, plot_metrics_from_file

def list_to_str(l):
    return '_'.join(map(str, l))


def is_in_range(label):
    return (label > 0.99) or (label < 1.e-6)
    # return (label < 1.e-6)


def main(measurement_subset):
    # load data
    batch_size = 512
    train_dataset = MeasurementDataset(root_path='./data/train/', measurement_subset=measurement_subset)
    train_dataset = FilteredDataset(train_dataset, filter_func=is_in_range, item_idx=1)
    
    test_dataset = MeasurementDataset(root_path='./data/val/', measurement_subset=measurement_subset)
    test_dataset = FilteredDataset(test_dataset, filter_func=is_in_range, item_idx=1)

    # train_dataset = VectorDensityMatrixDataset(root_path='./data/train/')
    # test_dataset = VectorDensityMatrixDataset(root_path='./data/val/')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if measurement_subset is None:
        model_name = 'regressor_filtered_data'
    else:
        model_name = f'regressor_m{list_to_str(measurement_subset)}'

    # create model
    model_save_path = f'./models/{model_name}.pt'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    model_params = {
    'input_dim': 16 if measurement_subset is None else len(measurement_subset),
    'output_dim': 1,
    'layers': 2,
    'hidden_size': 128,
    'input_dropout': 0.0
}
    # model = Regressor(**model_params)
    model = Regressor(**model_params)

    # train & test model
    log_path = f'./logs/{model_name}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    num_epochs = 40
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
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
    plot_metrics_from_file(log_path, title='Loss', save_path=f'./plots/{model_name}.png')


if __name__ == '__main__':
    # for measurement in range(16):
    #     main(measurement_subset=[measurement])
    main(measurement_subset=None)

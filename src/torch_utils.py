from tqdm import tqdm
import typing as t

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def train(
    model: nn.Module,
    device: torch.device, 
    train_loader: DataLoader, 
    optimizer: Optimizer, 
    epoch: int, 
    log_interval: int = 100, 
    criterion: t.Callable = nn.MSELoss()
) -> t.Dict[str, t.List[float]]:
    
    model.train()
    model.to(device)
    metrics = {'train_loss': 0}
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        metrics['train_loss'] += loss.item()
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    metrics['train_loss'] /= len(train_loader)
    return metrics


def test(
    model: nn.Module,
    device: torch.device, 
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable]
) -> t.Dict[str, t.List[float]]:
    
    model.eval()
    model.to(device)
    
    metrics = {name: 0 for name in criterions.keys()}
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing model...'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            for name, criterion in criterions.items():
                metrics[name] += criterion(output, target).item()
    for name in metrics.keys():
        metrics[name] /= len(test_loader)
        print(f'{name}: {metrics[name]:.4f}')
    return metrics


def regressor_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    input_threshold: float = 0.5,
    target_threshold: float = 0.5
) -> torch.Tensor:
    prediction = (input > input_threshold).float()
    target = (target > target_threshold).float()
    return (prediction == target).float().mean()
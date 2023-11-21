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
    criterion: t.Callable = nn.MSELoss()
) -> t.Dict[str, t.List[float]]:
    
    model.eval()
    model.to(device)
    
    metrics = {'test_loss': 0}
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing model...'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            metrics['test_loss'] += criterion(output, target).item()
    metrics['test_loss'] /= len(test_loader)
    print(f'Test set: Average loss: {metrics["test_loss"]:.4f}\n')
    return metrics
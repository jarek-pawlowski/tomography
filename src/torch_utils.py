from tqdm import tqdm
import typing as t
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from qiskit.quantum_info import DensityMatrix, state_fidelity

from src.utils_measure import Kwiat

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


def train_measurement_predictor(
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
    for batch_idx, (rho, measurement, _) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        optimizer.zero_grad()
        basis = torch.from_numpy(Kwiat.basis[0]).to(device).to(torch.complex64)
        basis = basis.unsqueeze(0).expand(rho.shape[0], -1, -1)
        measurement_with_basis = (measurement[:, 0:1], torch.stack((basis, basis), dim=1))
        predicted_rhos = model(measurement_with_basis, rho)
        loss = torch.zeros(1).to(device)
        for i in range(predicted_rhos.shape[1]):
            loss += criterion(predicted_rhos[:, i], rho)
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
    criterions: t.Dict[str, t.Callable],
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


def test_measurement_predictor(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    max_num_measurements: int = 16,
) -> t.Dict[str, t.List[float]]:
        
    model.eval()
    model.to(device)
    
    metrics = {name: {f'measurement {i}': 0 for i in range(max_num_measurements)} for name in criterions.keys()}
    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc='Testing model...'):
            rho, measurement = rho.to(device), measurement.to(device)
            basis = torch.from_numpy(Kwiat.basis[0]).to(device).to(torch.complex64)
            basis = basis.unsqueeze(0).expand(rho.shape[0], -1, -1)
            measurement_with_basis = (measurement[:, 0:1], torch.stack((basis, basis), dim=1))
            predicted_rhos = model(measurement_with_basis, rho)        
            for name, criterion in criterions.items():
                for i in range(predicted_rhos.shape[1]):
                    metrics[name][f'measurement {i}'] += criterion(predicted_rhos[:, i], rho).item()
    for name in metrics.keys():
        for i in range(max_num_measurements):
            metrics[name][f'measurement {i}'] /= len(test_loader)
            print(f'{name} - measurement {i}: {metrics[name][f"measurement {i}"]:.4f}')
    return metrics


def test_varying_input(
    model: nn.Module,
    device: torch.device, 
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    varying_input_idx: t.Optional[t.List[int]],
    max_variance: float = 1.,
    step: float = 0.1,
) -> t.Dict[str, t.List[float]]:
    
    model.eval()
    model.to(device)
    
    metrics = {}
    for variance in np.arange(0, max_variance, step):
        metrics[variance] = {name: 0 for name in criterions.keys()}
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f' Variance: {variance}'):
                data, target = data.to(device), target.to(device)
                data_min = torch.maximum(data[:, torch.tensor(varying_input_idx)] - variance, torch.zeros_like(data[:, torch.tensor(varying_input_idx)]))
                data_max = torch.minimum(data[:, torch.tensor(varying_input_idx)] + variance, torch.ones_like(data[:, torch.tensor(varying_input_idx)]))
                interval = data_max - data_min + 1e-6
                varied_data = torch.rand_like(interval) * interval + data_min
                data[:, torch.tensor(varying_input_idx)] = varied_data
                output = model(data)
                for name, criterion in criterions.items():
                    metrics[variance][name] += ((criterion(output, target) * interval).sum() / interval.sum()).item()
        for name in metrics[variance].keys():
            metrics[variance][name] /= len(test_loader)
            print(f'{name} - variance {variance}: {metrics[variance][name]:.4f}')
    return metrics


def regressor_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    input_threshold: float = 0.5,
    target_threshold: float = 0.5,
    reduction: str = 'mean'
) -> torch.Tensor:
    prediction = (input > input_threshold).float()
    target = (target > target_threshold).float()
    accuracy = (prediction == target).float()
    if reduction == 'mean':
        return accuracy.mean()
    return accuracy


def reduced_input_criterion(
    input: torch.Tensor,
    target: torch.Tensor,
    input_drop_idx: t.List[int],
    criterion: t.Callable
) -> torch.Tensor:
    input = torch.cat([input[:, :input_drop_idx[0]], input[:, input_drop_idx[0] + 1:]], dim=1)
    for idx in input_drop_idx[1:]:
        input = torch.cat([input[:, :idx], input[:, idx + 1:]], dim=1)
    return criterion(input, target)


def torch_bures_distance(
    rho_input: torch.Tensor,
    rho_target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    rho_input_np = torch.complex(rho_input[..., 0, :, :], rho_input[..., 1, :, :]).to(torch.cdouble)
    rho_target_np = torch.complex(rho_target[..., 0, :, :], rho_target[..., 1, :, :]).to(torch.cdouble)

    bures_distances = []
    for rho_in, rho_t in zip(rho_input_np, rho_target_np):
        fidelity = torch_fidelity(rho_in, rho_t)
        bures_distance = 2 * (1 - torch.sqrt(fidelity))
        bures_distances.append(bures_distance.unsqueeze(0))
    bures_distances = torch.stack(bures_distances)
    if reduction == 'mean':
        return bures_distances.mean()
    return bures_distances


def torch_fidelity(
    rho1 : torch.Tensor,
    rho2: torch.Tensor
):
    unitary1, singular_values, unitary2 = torch.linalg.svd(rho1)
    diag_func_singular = torch.diag(torch.sqrt(singular_values)).to(torch.cdouble)
    s1sqrt =  unitary1.matmul(diag_func_singular).matmul(unitary2)   

    unitary1, singular_values, unitary2 = torch.linalg.svd(rho2)
    diag_func_singular = torch.diag(torch.sqrt(singular_values)).to(torch.cdouble)
    s2sqrt =  unitary1.matmul(diag_func_singular).matmul(unitary2)   

    fid = torch.linalg.norm(s1sqrt.matmul(s2sqrt), ord="nuc") ** 2
    return fid.to(torch.double)
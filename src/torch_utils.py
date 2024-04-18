from collections import defaultdict
from itertools import product
from functools import reduce
from tqdm import tqdm
import typing as t
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.distributions.multivariate_normal import MultivariateNormal
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
    criterion: t.Callable = nn.MSELoss(),
    bases_loss_fn: t.Optional[t.Callable] = None
) -> t.Dict[str, t.List[float]]:
    
    model.train()
    model.to(device)
    metrics = {'train_loss': 0, 'bases_loss': 0}
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (rho, measurement, _) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        optimizer.zero_grad()
        basis = torch.from_numpy(Kwiat.basis[0]).to(device).to(torch.complex64)
        basis = basis.unsqueeze(0).expand(rho.shape[0], -1, -1)
        measurement_with_basis = (measurement[:, 0:1], torch.stack((basis, basis), dim=1))
        predicted_rhos, predicted_bases = model(measurement_with_basis, rho)
        loss = torch.zeros(1).to(device)
        for i in range(predicted_rhos.shape[1]):
            loss += criterion(predicted_rhos[:, i], rho)
        if bases_loss_fn is not None:
            bases_loss = bases_loss_fn(predicted_bases)
            metrics['bases_loss'] += bases_loss.item()
            loss += bases_loss
        loss.backward()
        optimizer.step()
        metrics['train_loss'] += loss.item()
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    metrics['train_loss'] /= len(train_loader)
    metrics['bases_loss'] /= len(train_loader)
    return metrics


def train_discrete_measurement_selector(
    model: nn.Module,
    device: torch.device, 
    train_loader: DataLoader, 
    optimizer: Optimizer, 
    epoch: int, 
    log_interval: int = 100, 
    reconstructor_criterion: t.Callable = nn.MSELoss(),
    selector_criterion: t.Callable = nn.MSELoss(reduction='none'),
) -> t.Dict[str, t.List[float]]:
    
    model.train()
    model.to(device)
    metrics = {'reconstructor_train_loss': 0, 'selector_train_loss': 0}
    bases = [
        torch.from_numpy(base).to(device).to(torch.complex64)
        for base in Kwiat.basis
    ]
    qubits_bases = [torch.stack(multi_qubit_base) for multi_qubit_base in product(bases, repeat=model.num_qubits)]
    qubits_bases = torch.stack(qubits_bases)
    qubits_bases = qubits_bases.unsqueeze(0).expand(train_loader.batch_size, -1, -1, -1, -1)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (rho, measurement, _) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        optimizer.zero_grad()

        measurement_with_basis = [
            (measurement[:, i:i+1], qubits_bases[:, i])
            for i in range(measurement.shape[1])
        ]
        predicted_best_rhos, predicted_bases_probabilities, predicted_all_rhos = model(measurement_with_basis, rho)
        reconstructor_loss = torch.zeros(1).to(device)
        selector_loss = torch.zeros(1).to(device)
        for i in range(predicted_best_rhos.shape[1]):
            reconstructor_loss += reconstructor_criterion(predicted_best_rhos[:, i], rho)
            probabilites = reduce(torch.func.vmap(torch.kron), [predicted_bases_probabilities[:, i, j] for j in range(predicted_bases_probabilities.shape[2])])
            selector_loss_i = probabilites.view(-1, probabilites.shape[1], 1, 1, 1) * selector_criterion(predicted_all_rhos[:, i], rho.unsqueeze(1).expand(-1, predicted_all_rhos.shape[2], *rho.shape[1:])).detach()
            selector_loss += selector_loss_i.sum(dim=1).mean()

        reconstructor_loss.backward()
        selector_loss.backward()
        optimizer.step()
        metrics['reconstructor_train_loss'] += reconstructor_loss.item()
        metrics['selector_train_loss'] += selector_loss.item()
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'reconstructor_loss': reconstructor_loss.item(), 'selector_loss': selector_loss.item()})
    metrics['reconstructor_train_loss'] /= len(train_loader)
    metrics['selector_train_loss'] /= len(train_loader)
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
            predicted_rhos, predicted_bases = model(measurement_with_basis, rho)        
            for name, criterion in criterions.items():
                for i in range(predicted_rhos.shape[1]):
                    metrics[name][f'measurement {i}'] += criterion(predicted_rhos[:, i], rho).item()
    for name in metrics.keys():
        for i in range(max_num_measurements):
            metrics[name][f'measurement {i}'] /= len(test_loader)
            print(f'{name} - measurement {i}: {metrics[name][f"measurement {i}"]:.4f}')
    return metrics


def test_discrete_measurement_selector(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    max_num_measurements: int = 16,
) -> t.Dict[str, t.List[float]]:
        
    model.eval()
    model.to(device)

    bases = [
        torch.from_numpy(base).to(device).to(torch.complex64)
        for base in Kwiat.basis
    ]
    qubits_bases = [torch.stack(multi_qubit_base) for multi_qubit_base in product(bases, repeat=model.num_qubits)]
    qubits_bases = torch.stack(qubits_bases)
    qubits_bases = qubits_bases.unsqueeze(0).expand(test_loader.batch_size, -1, -1, -1, -1)
    
    metrics = {name: {f'measurement {i}': 0 for i in range(max_num_measurements)} for name in criterions.keys()}
    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc='Testing model...'):
            rho, measurement = rho.to(device), measurement.to(device)
            measurement_with_basis = [
                (measurement[:, i:i+1], qubits_bases[:, i])
                for i in range(measurement.shape[1])
            ]
            predicted_best_rhos, predicted_bases_probabilities, predicted_all_rhos = model(measurement_with_basis, rho)
            for name, criterion in criterions.items():
                for i in range(predicted_best_rhos.shape[1]):
                    metrics[name][f'measurement {i}'] += criterion(predicted_best_rhos[:, i], rho).item()
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


def test_varying_feature(
    model: t.Union[nn.Module, t.Callable],
    device: torch.device, 
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    feature_idx: t.Optional[t.List[int]],
    feature_value_range: t.Tuple[int, int] = (0., 1.),
    step: float = 0.1,
    model_output_mean: t.Optional[torch.Tensor] = None,
) -> t.Dict[str, t.List[float]]:
    
    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)
    
    avg_outputs = defaultdict(float)
    avg_distances = defaultdict(float)
    for feature_value in np.arange(*feature_value_range, step):
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f' Feature value: {feature_value}'):
                data, target = data.to(device), target.to(device)
                data[:, torch.tensor(feature_idx)] = feature_value
                output = model(data)
                if model_output_mean is not None:
                    avg_distances[feature_value] += torch.abs(output - model_output_mean).mean().item()
                avg_outputs[feature_value] += output.mean().item()
        avg_outputs[feature_value] /= len(test_loader)
        avg_distances[feature_value] /= len(test_loader)

    outputs = torch.tensor(list(avg_outputs.values()))
    mean_metrics = {
        criterion_name: criterion(outputs) for criterion_name, criterion in criterions.items()
    }
    distances = torch.tensor(list(avg_distances.values()))
    distance_metrics = {
        criterion_name: criterion(distances) for criterion_name, criterion in criterions.items()
    }
    return mean_metrics, distance_metrics


def test_output_statistics_for_given_feature(
    model: t.Union[nn.Module, t.Callable],
    device: torch.device, 
    test_loader: DataLoader,
    feature_idx: t.Optional[t.List[int]],
    criterions: t.Dict[str, t.Callable],
    model_output_mean: t.Optional[torch.Tensor] = None
) -> t.Dict[str, t.List[float]]:
    
    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)
    
    outputs = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc=f' Testing model...'):
            data = data.to(device)
            feature_data = data[:, torch.tensor(feature_idx)]
            # masked_data = torch.zeros_like(data)
            masked_data = torch.rand_like(data)
            masked_data[:, torch.tensor(feature_idx)] = feature_data
            output = model(masked_data)
            if model_output_mean is not None:
                output -= model_output_mean
            outputs.append(output)
    outputs = torch.cat(outputs)
    metrics = {
        criterion_name: criterion(outputs) for criterion_name, criterion in criterions.items()
    }
    return metrics


def test_output_statistics_varying_feature(
    model: t.Union[nn.Module, t.Callable],
    device: torch.device, 
    feature_idx: t.Optional[t.List[int]],
    criterions: t.Dict[str, t.Callable],
    features_num: int = 16,
    feature_value_range: t.Tuple[int, int] = (0., 1.),
    step: float = 0.1, 
    mean: t.Optional[torch.Tensor] = None,
    covariance_matrix: t.Optional[torch.Tensor] = None,
    model_output_mean: t.Optional[torch.Tensor] = None
) -> t.Dict[str, t.List[float]]:
    
    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)
    
    outputs = []
    for feature_value in tqdm(np.arange(*feature_value_range, step), desc=f' Varying feature...'):
        with torch.no_grad():
            data = torch.zeros(1, features_num).to(device)
            # generate data from multivariate normal distribution
            if mean is not None and covariance_matrix is not None:
                data = generate_sample_from_mean_and_covariance(mean, covariance_matrix, batch_size=100)
            data[:, torch.tensor(feature_idx)] = feature_value
            output = model(data)
            if model_output_mean is not None:
                output -= model_output_mean
            outputs.append(torch.mean(output))
    # outputs = torch.cat(outputs)
    outputs = torch.tensor(outputs)
    metrics = {
        criterion_name: criterion(outputs) for criterion_name, criterion in criterions.items()
    }
    return metrics


def calculate_mean_model_output_with_varied_feature(
    model: t.Union[nn.Module, t.Callable],
    device: torch.device,
    test_loader: DataLoader,
    features_value_range: t.Tuple[int, int] = (0., 1.),
    step: float = 0.1, 
) -> float:
    
    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)
    
    avg_output = 0.
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc=f' Calculating average model output...'):
            data = data.to(device)
            num_features = data.shape[1]
            for feature_idx in range(num_features):
                for feature_value in np.arange(*features_value_range, step):
                    data[:, torch.tensor([feature_idx])] = feature_value
                    output = model(data)
                    avg_output += output.sum().item()
    avg_output /= (len(test_loader.dataset) * len(np.arange(*features_value_range, step)) * num_features)
    return avg_output



def calculate_dataset_statistics(
    data_loader: DataLoader,
    device: torch.device,
    callable: t.Optional[t.Callable] = None
):
    # calculate latent space distribution (mean and std)
    population = []
    for data, _ in tqdm(data_loader, 'Collecting data statistics...'):
        data = data.to(device)
        if callable is not None:
            data = callable(data)
        population.append(data)
    population = torch.cat(population, dim=0)
    statistics = {
        'mean': torch.mean(population, dim=0),
        'std': torch.std(population, dim=0),
        'covariance_matrix': torch.cov(population.T),
        'max': torch.max(population, dim=0).values,
        'min': torch.min(population, dim=0).values
    }
    return statistics


def generate_sample_from_mean_and_covariance(mean: torch.Tensor, covariance_matrix: torch.Tensor, batch_size: int = 1):
    mvn = MultivariateNormal(mean, covariance_matrix)
    return mvn.sample((batch_size,))


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
    rho1: torch.Tensor,
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


def bases_loss(
    predicted_bases: torch.Tensor, # shape (batch_size, num_measurements, num_qubits, 2, 2)
    target_bases: torch.Tensor, # shape (num_target_bases, 2, 2)
    reduction: str = 'mean'
) -> torch.Tensor:
    predicted_bases_complex_stack = torch.stack((predicted_bases.real, predicted_bases.imag), dim=-3)
    bases_loss = []
    for target_base in target_bases:
        target_base = target_base.view(1, 1, 1, 2, 2).expand(predicted_bases.shape[0], predicted_bases.shape[1], predicted_bases.shape[2], -1, -1).to(predicted_bases.device)
        target_base_complex_stack = torch.stack((target_base.real, target_base.imag), dim=-3)
        base_loss = torch.nn.functional.mse_loss(predicted_bases_complex_stack, target_base_complex_stack, reduction='none').mean(dim=(-1, -2, -3))
        bases_loss.append(base_loss)
    bases_loss = torch.stack(bases_loss) # shape (num_target_bases, batch_size, num_measurements, num_qubits)
    bases_loss = torch.min(bases_loss, dim=0).values

    if reduction == 'mean':
        return bases_loss.mean()
    return bases_loss


def reconstruct_rho(
    model: nn.Module,
    device: torch.device,
    rho: torch.Tensor,
    measurement: torch.Tensor,
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
    model.eval()
    model.to(device)

    basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat.basis]
    
    with torch.no_grad():
        rho = rho.to(device).unsqueeze(0)
        measurement = measurement.to(device).unsqueeze(0)
        basis = torch.from_numpy(Kwiat.basis[0]).to(device).to(torch.complex64)
        basis = basis.unsqueeze(0)
        measurement_with_basis = (measurement[:, 0:1], torch.stack((basis, basis), dim=1))
        predicted_rhos, predicted_bases = model(measurement_with_basis, rho)        
        # It has to be flatten first
        predicted_bases_flatten = torch.flatten(predicted_bases, start_dim=0, end_dim=-3).view(-1, 1, 4)
        basis_matrices_flatten = torch.stack(basis_matrices).view(1, 4, 4).expand(predicted_bases.shape[0], -1, -1).to(device)
        
        bases_distances_real = torch.cdist(predicted_bases_flatten.real, basis_matrices_flatten.real, p=2)
        bases_distances_imag = torch.cdist(predicted_bases_flatten.imag, basis_matrices_flatten.imag, p=2)
        bases_distances = bases_distances_real + bases_distances_imag
        # Take basis with minimal distance
        predicted_base_idx = torch.argmin(bases_distances, dim=-1)
        predicted_base_idx = torch.unflatten(predicted_base_idx, dim=0, sizes=(predicted_bases.shape[1], predicted_bases.shape[2]))
        rho_diff = torch.abs(predicted_rhos - rho.unsqueeze(1))
    return predicted_rhos.squeeze(), rho_diff.squeeze(), predicted_base_idx.squeeze()
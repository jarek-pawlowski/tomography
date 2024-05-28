from collections import defaultdict
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
from src.torch_measure import tensordot

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
    no_qubits: int,
    log_interval: int = 100, 
    criterion: t.Callable = nn.MSELoss(),
    bases_loss_fn: t.Optional[t.Callable] = None,
) -> t.Dict[str, t.List[float]]:
    
    model.train()
    model.to(device)
    metrics = {'train_loss': 0, 'bases_loss': 0}
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (rho, measurement, _) in pbar:
        #print(rho.shape[-1], rho)
        #no_qubits = np.log2(rho.shape[-1]).astype(int)
        #no_qubits = int((rho.shape[-1]))
        rho, measurement = rho.to(device), measurement.to(device)
        basis_comp_vector = torch.Tensor([[1,0,0]]*no_qubits).to(device)
        snapshot_batch = []
        # make initial measurements
        # iterate over batch (to be paralelized!)
        for rho_k in rho:
            #breakpoint()
            rho_k = torch.complex(rho_k[0], rho_k[1]).view(*[2]*no_qubits)
            #print(rho_k)

            snapshot_batch.append(model.take_snapshot(rho_k, basis_comp_vector))
        initial_snapshot_with_basis = (torch.Tensor(snapshot_batch).to(device), basis_comp_vector.expand(rho.shape[0], -1, -1))
        optimizer.zero_grad()
        predicted_rhos, predicted_bases = model(initial_snapshot_with_basis, rho) #here model runs
        loss = torch.zeros(1).to(device)
        
        for pred, gt in zip(predicted_rhos, rho): #loss over all predicted rhos gt is loaded from dataset
            #print(rho.shape, predicted_rhos[i].shape)
            #print(f"Tensor to check: {predicted_rhos[i]}")
            psi_gt = torch.complex(gt[0], gt[1]).view(*[2]*no_qubits).reshape(int(pow(2, no_qubits)))
            
            rho0  = tensordot(psi_gt, psi_gt, indices=0, conj_tr=(False,True)) #density matrix torch tensordot changed 2nd True
            #rho0 = rho0_bez.reshape(int(pow(2, no_qubits)),int(pow(2, no_qubits)))
            rho0_real = rho0.real
            rho0_imag = rho0.imag
            stacked_tensor = torch.stack([rho0_real, rho0_imag], dim=0)
            
            stacked_tensor = stacked_tensor.unsqueeze(0)  # Add an extra dimension at the beginning
            
            #print(f"Orginal tensor: {stacked_tensor}")
            #print(f"Predicted tensor: {pred}")
            
            loss += criterion(pred, stacked_tensor)
        # enforce to select only selected Pauli as basis: ???
        # if bases_loss_fn is not None:
        #     bases_loss = bases_loss_fn(predicted_bases)
        #     metrics['bases_loss'] += bases_loss.item()
        #     loss += bases_loss
        loss.backward()
        optimizer.step()
        metrics['train_loss'] += loss.item()
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    metrics['train_loss'] /= len(train_loader)
    metrics['bases_loss'] /= len(train_loader)
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
    no_qubits: int,
    criterions: t.Dict[str, t.Callable],
    num_returned_reconstructions: int = 1,
) -> t.Dict[str, t.List[float]]:
        
    model.eval()
    model.to(device)
    
    metrics = {name: {f'reconstruction {i}': 0 for i in range(num_returned_reconstructions)} for name in criterions.keys()}
    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc='Testing model...'):
            #no_qubits = np.log2(rho.shape[-1]).astype(int)
            #no_qubits = int(rho.shape[-1])
            rho, measurement = rho.to(device), measurement.to(device)
            basis_comp_vector = torch.Tensor([[1,0,0]]*no_qubits).to(device)
            # iterate over batch (to be paralelized!)
            snapshot_batch = []
            for rho_k in rho:
                rho_k = torch.complex(rho_k[0], rho_k[1]).view(*[2]*no_qubits)
                snapshot_batch.append(model.take_snapshot(rho_k, basis_comp_vector))
            initial_snapshot_with_basis = (torch.Tensor(snapshot_batch).to(device), basis_comp_vector.expand(rho.shape[0], -1, -1))
            predicted_rhos, _ = model(initial_snapshot_with_basis, rho)        
            for name, criterion in criterions.items():
                for i, (pred, gt) in enumerate(zip(predicted_rhos, rho)): #loss over all predicted rhos gt is loaded from dataset
                    psi_gt = torch.complex(gt[0], gt[1]).view(*[2]*no_qubits).reshape(int(pow(2, no_qubits)))
                    rho0  = tensordot(psi_gt, psi_gt, indices=0, conj_tr=(False,True)) #density matrix torch tensordot changed 2nd True
                    rho0_real = rho0.real
                    rho0_imag = rho0.imag
                    stacked_tensor = torch.stack([rho0_real, rho0_imag], dim=0)
                    
                    stacked_tensor = stacked_tensor.unsqueeze(0)  # Add an extra dimension at the beginning

                    metrics[name][f'reconstruction {i}'] += criterion(pred, stacked_tensor).item()
    for name in metrics.keys():
        for i in range(num_returned_reconstructions):
            metrics[name][f'reconstruction {i}'] /= len(test_loader)
            print(f'{name} - reconstruction {i}: {metrics[name][f"reconstruction {i}"]:.4f}')
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

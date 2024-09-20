import random
import typing as t
from functools import reduce
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils import generate_mean_sample
from src.tomography_utils_torch import calculate_concurrence_from_measurements, reconstruct
from src.tomography_utils_numpy import N_QUBIT_GAMMAS, Kwiat_library, Kwiat_projectors, basis_for_Kwiat_code
from src.hlp import reconstruct_1qbit_hlp


def test_1qbit_hlp(
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    measurements_subset: t.Optional[t.Union[int, t.List[int]]] = None,
    device: torch.device  = torch.device('cpu'),
) -> t.Dict[str, t.List[float]]:
    num_qubits = test_loader.dataset.num_qubits
    assert num_qubits == 1, 'HLP defined only for 1 qubit'

    metrics = {name: 0 for name in criterions.keys()}

    with torch.no_grad():
        for rho, measurements, _ in tqdm(test_loader, desc='Testing model...'):
            rho, measurements = rho.to(device), measurements.to(device)
            if type(measurements_subset) == int:
                measurements_subset = random.sample(range(measurements.shape[1]), measurements_subset)
            if measurements_subset is not None:
                measurements = measurements[:, measurements_subset]
            else:
                measurements_subset = list(range(measurements.shape[1]))

            reconstructed_rho = reconstruct_1qbit_hlp(measurements, measurements_subset)
            reconstructed_rho = torch.stack([reconstructed_rho.real, reconstructed_rho.imag], dim=1)

            for name, criterion in criterions.items():
                metrics[name] += criterion(reconstructed_rho, rho)
    for name in metrics.keys():
        metrics[name] /= len(test_loader)
        try:
            print(f'{name}: {metrics[name]:.4f}')
            metrics[name] = metrics[name].item()
        except:
            pass
    return metrics


def test_mean_reconstruction(
    mean_rho: torch.Tensor,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    device: torch.device = torch.device('cpu'),
) -> t.Dict[str, t.List[float]]:
    
    metrics = {name: 0 for name in criterions.keys()}
    with torch.no_grad():
        for rho, _, _ in tqdm(test_loader, desc='Testing model...'):
            rho = rho.to(device)
            for name, criterion in criterions.items():
                metrics[name] += criterion(mean_rho.unsqueeze(0).expand_as(rho), rho)
    for name in metrics.keys():
        metrics[name] /= len(test_loader)
        try:
            print(f'{name}: {metrics[name]:.4f}')
            metrics[name] = metrics[name].item()
        except:
            pass
    return metrics
        


def test_kwiat_gammas_reconstruction(
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    measurements_subset: t.Optional[t.Union[int, t.List[int]]] = None,
    inverse: str = 'pinv',
    enforce_valid_density_matrix: bool = False,
) -> t.Dict[str, t.List[float]]:

    num_qubits = test_loader.dataset.num_qubits

    metrics = {name: 0 for name in criterions.keys()}

    single_qubits_projection_vectors = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat_projectors.basis]
    # two_qubits_projection_vectors = torch.stack([torch.kron(basis1, basis2) for basis1, basis2 in product(single_qubits_projection_vectors, repeat=2)])
    n_qubits_projection_vectors = torch.stack([reduce(torch.kron, [basis_i for basis_i in basis]) for basis in product(single_qubits_projection_vectors, repeat=num_qubits)])
    selected_projection_vectors = n_qubits_projection_vectors

    # gammas = torch.tensor(Gammas, dtype=torch.complex64, device=device)
    # gammas = torch.func.vmap(torch.kron)(two_qubits_basis_matrices[:, 0], two_qubits_basis_matrices[:, 1])
    gammas = torch.tensor(N_QUBIT_GAMMAS(num_qubits), dtype=torch.complex64, device=device)
    selected_gammas = gammas

    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc='Testing model...'):
            rho, measurement = rho.to(device), measurement.to(device)
            if type(measurements_subset) == int:
                measurements_subset = random.sample(range(measurement.shape[1]), measurements_subset)
            if measurements_subset is not None:
                measurement = measurement[:, measurements_subset]
                selected_projection_vectors = n_qubits_projection_vectors[measurements_subset]
                if inverse != 'pinv':
                    selected_gammas = gammas[measurements_subset]

            reconstructed_rho = torch.stack([reconstruct(measurement_i, selected_projection_vectors, selected_gammas, enforce_valid_density_matrix=enforce_valid_density_matrix, inverse=inverse) for measurement_i in measurement])
            reconstructed_rho = torch.stack([reconstructed_rho.real, reconstructed_rho.imag], dim=1)

            for name, criterion in criterions.items():
                metrics[name] += criterion(reconstructed_rho, rho)
    for name in metrics.keys():
        metrics[name] /= len(test_loader)
        try:
            print(f'{name}: {metrics[name]:.4f}')
            metrics[name] = metrics[name].item()
        except:
            pass
    return metrics


def test_reconstruction_from_noisy_measurements(
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    varying_input_idx: t.Optional[t.List[int]],
    noise_variance: float,
    strategy: str = 'tomography',
    method: str = 'MLE',
    use_intensity: bool = False, # param effective for 'optimized_tomography' strategy
) -> t.Dict[str, t.List[float]]:

    variance_metrics = {name: 0 for name in criterions.keys()}

    num_qubits = test_loader.dataset.num_qubits

    single_qubits_projection_vectors = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat_projectors.basis]
    n_qubits_projection_vectors = torch.stack([reduce(torch.kron, [basis_i for basis_i in basis]) for basis in product(single_qubits_projection_vectors, repeat=num_qubits)])

    gammas = torch.tensor(N_QUBIT_GAMMAS(num_qubits), dtype=torch.complex64)

    optimized_tomography = Kwiat_library(basis_for_Kwiat_code)

    with torch.no_grad():
        for rho, measurements, _ in tqdm(test_loader, desc=f' Variance: {noise_variance}'):
            if varying_input_idx is not None:
                data_min = torch.maximum(measurements[:, torch.tensor(varying_input_idx)] - noise_variance, torch.zeros_like(measurements[:, torch.tensor(varying_input_idx)]))
                data_max = torch.minimum(measurements[:, torch.tensor(varying_input_idx)] + noise_variance, torch.ones_like(measurements[:, torch.tensor(varying_input_idx)]))
                interval = data_max - data_min + 1e-6
                varied_data = torch.rand_like(interval) * interval + data_min
                measurements[:, torch.tensor(varying_input_idx)] = varied_data

            predictions = []
            for measurement in measurements:
                if strategy == 'tomography':
                    if (method == 'zeroed_measurements') and (varying_input_idx is not None):
                        zero_measurements = varying_input_idx
                    elif (method == 'random_measurements') or (varying_input_idx is None):
                        zero_measurements = None
                    else:
                        raise ValueError(f'Unknown method for tomography: {method}')
                    rho_rec = reconstruct(measurement, n_qubits_projection_vectors, gammas, enforce_valid_density_matrix=False, zero_measurements=zero_measurements).cpu().numpy()
                elif strategy == 'optimized_tomography':
                    intensity = None
                    if use_intensity:
                        intensity = np.ones(len(measurement))
                        if varying_input_idx is not None:
                            intensity[varying_input_idx] = 1 - noise_variance + 1e-6
                    rho_rec = optimized_tomography.run_tomography(measurement.numpy(), method=method, intensity=intensity)
                else:
                    raise ValueError(f'Unknown strategy: {strategy}')
                matrix_r = np.real(rho_rec)
                matrix_im = np.imag(rho_rec)
                rho_rec_t = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0)).float()
                predictions.append(rho_rec_t)

            predictions = torch.stack(predictions)
            weights = torch.ones(predictions.shape[0]).to(predictions.device)
            if varying_input_idx is not None:
                weights = interval.mean(dim=-1) # averaging interval for all disturbed measurements
            for name, criterion in criterions.items():
                error = torch.flatten(criterion(predictions, rho), start_dim=1, end_dim=-1).mean(dim=-1)
                variance_metrics[name] += ((error * weights).sum() / weights.sum()).item() / len(test_loader)

    return variance_metrics


def test_concurrence_measurement_noise_for_variance(
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    varying_input_idx: t.Optional[t.List[int]],
    variance: float,
) -> t.Dict[str, t.List[float]]:

    variance_metrics = {name: 0 for name in criterions.keys()}
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f' Variance: {variance}'):
            data_min = torch.maximum(data[:, torch.tensor(varying_input_idx)] - variance, torch.zeros_like(data[:, torch.tensor(varying_input_idx)]))
            data_max = torch.minimum(data[:, torch.tensor(varying_input_idx)] + variance, torch.ones_like(data[:, torch.tensor(varying_input_idx)]))
            interval = data_max - data_min + 1e-6
            varied_data = torch.rand_like(interval) * interval + data_min
            data[:, torch.tensor(varying_input_idx)] = varied_data
            predictions = calculate_concurrence_from_measurements(data)
            interval[predictions.squeeze() == -1] = 0

            for name, criterion in criterions.items():

                variance_metrics[name] += ((criterion(predictions, target) * interval).sum() / interval.sum()).item() / len(test_loader)

    return variance_metrics


def test_concurrence_measurement_noise(
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    varying_input_idx: t.Optional[t.List[int]],
    max_variance: float = 1.,
    step: float = 0.1,
) -> t.Dict[str, t.Dict[str, t.List[float]]]:

    metrics = {}
    for variance in np.arange(0, max_variance, step):
        metrics[variance] = test_concurrence_measurement_noise_for_variance(test_loader, criterions, varying_input_idx, variance)
        for name in metrics[variance].keys():
            print(f'{name} - variance {variance}: {metrics[variance][name]:.4f}')
    return metrics
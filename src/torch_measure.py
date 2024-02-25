from math import log2
import typing as t
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from qiskit.quantum_info import concurrence, DensityMatrix

from src.utils_measure import Tomography, Kwiat_projectors, Kwiat_library, basis_for_Kwiat_code


def tensordot(a: torch.Tensor, b: torch.Tensor, indices: t.Tuple[t.List[int], t.List[int]] = ([1], [0]), moveaxis=None):
    result = torch.tensordot(a, b, indices)
    if moveaxis is not None:
        result = torch.moveaxis(result, *moveaxis)
    return result

def trace(a: torch.Tensor):
    # performs tensor contraction Tijk...ijk...
    dim = int(len(a.shape)/2)
    indices = np.indices([2]*dim).reshape(dim,-1).T
    indices_to_sum = np.tile(indices, 2)
    return torch.sum(torch.stack([a[tuple(idx)] for idx in indices_to_sum]))

def measure(rho: torch.Tensor, basis_vectors: t.Tuple[torch.Tensor, torch.Tensor]):
    # measure all qubits using list of operators
    # basis_vectors = operators to use when measuring subsequent qubits
    
    Prho = rho.clone()
    for i, basis_vector in enumerate(basis_vectors):
        Prho = tensordot(basis_vector, Prho, indices=([1], [i]), moveaxis=(0,i))
    prob = trace(Prho).real
    return prob


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


def test_concurrence_measurement_noise_for_variance(
    test_loader: DataLoader, 
    criterions: t.Dict[str, t.Callable],
    varying_input_idx: t.Optional[t.List[int]],
    variance: float,
) -> t.Dict[str, t.List[float]]:
    
    variance_metrics = {name: 0 for name in criterions.keys()}
    num_skipped = 0

    num_qubits = 2 # tested for 2 qubits
    dim = 2**num_qubits
    
    tomography = Tomography(num_qubits, Kwiat_projectors)
    tomography.calulate_B_inv()

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f' Variance: {variance}'):
            data_min = torch.maximum(data[:, torch.tensor(varying_input_idx)] - variance, torch.zeros_like(data[:, torch.tensor(varying_input_idx)]))
            data_max = torch.minimum(data[:, torch.tensor(varying_input_idx)] + variance, torch.ones_like(data[:, torch.tensor(varying_input_idx)]))
            interval = data_max - data_min + 1e-6
            varied_data = torch.rand_like(interval) * interval + data_min
            data[:, torch.tensor(varying_input_idx)] = varied_data

            predictions = []
            for measurements in data:
                rho_rec = tomography.reconstruct(measurements.numpy(), enforce_positiv_sem=True)
                rho_rec = rho_rec.reshape((dim, dim))
                rho_rec = rho_rec / np.trace(rho_rec)
                density_matrix = DensityMatrix(rho_rec)
                try:
                    conc = concurrence(density_matrix)
                except:
                    conc = -1
                    num_skipped += 1
                predictions.append(conc)
            predictions = torch.tensor(predictions).unsqueeze(-1)

            interval[predictions.squeeze() == -1] = 0

            for name, criterion in criterions.items():

                variance_metrics[name] += ((criterion(predictions, target) * interval).sum() / interval.sum()).item() / len(test_loader)

    print(f'Percentage of skipped samples (due to incorrect reconstruction): {num_skipped/len(test_loader.dataset)*100:.2f}')

    return variance_metrics


def test_reconstruction_measurement_noise_for_variance(
    test_loader: DataLoader, 
    criterions: t.Dict[str, t.Callable],
    varying_input_idx: t.Optional[t.List[int]],
    variance: float,
    strategy: str = 'tomography',
    method: str = 'MLE', # param effective for 'optimized_tomography' strategy
    use_intensity: bool = False # param effective for 'optimized_tomography' strategy
) -> t.Dict[str, t.List[float]]:
    
    variance_metrics = {name: 0 for name in criterions.keys()}

    num_qubits = 2 # tested for 2 qubits
    dim = 2**num_qubits
    
    tomography = Tomography(num_qubits, Kwiat_projectors)
    tomography.calulate_B_inv()

    optimized_tomography = Kwiat_library(basis_for_Kwiat_code)

    with torch.no_grad():
        for rho, measurements, _ in tqdm(test_loader, desc=f' Variance: {variance}'):
            data_min = torch.maximum(measurements[:, torch.tensor(varying_input_idx)] - variance, torch.zeros_like(measurements[:, torch.tensor(varying_input_idx)]))
            data_max = torch.minimum(measurements[:, torch.tensor(varying_input_idx)] + variance, torch.ones_like(measurements[:, torch.tensor(varying_input_idx)]))
            interval = data_max - data_min + 1e-6
            varied_data = torch.rand_like(interval) * interval + data_min
            measurements[:, torch.tensor(varying_input_idx)] = varied_data

            predictions = []
            for measurement in measurements:
                if strategy == 'tomography':
                    rho_rec = tomography.reconstruct(measurement.numpy(), enforce_positiv_sem=True)
                elif strategy == 'optimized_tomography':
                    intensity = None
                    if use_intensity:
                        intensity = np.ones(len(measurement))
                        intensity[varying_input_idx] = 1 - variance + 1e-6
                    rho_rec = optimized_tomography.run_tomography(measurement.numpy(), method=method, intensity=intensity)
                else:
                    raise ValueError(f'Unknown strategy: {strategy}')
                rho_rec = rho_rec.reshape((dim, dim))
                rho_rec = rho_rec / np.trace(rho_rec)
                matrix_r = np.real(rho_rec)
                matrix_im = np.imag(rho_rec)
                rho_rec_t = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0)).float()
                predictions.append(rho_rec_t)

            predictions = torch.stack(predictions)
            weights = interval.mean(dim=-1) # averaging interval for all disturbed measurements
            for name, criterion in criterions.items():
                error = torch.flatten(criterion(predictions, rho), start_dim=1, end_dim=-1).mean(dim=-1)
                variance_metrics[name] += ((error * weights).sum() / weights.sum()).item() / len(test_loader)

    return variance_metrics

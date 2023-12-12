from math import log2
import typing as t
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from qiskit.quantum_info import concurrence, DensityMatrix

from src.utils_measure import Tomography, Kwiat_projectors


def tensordot(a: torch.Tensor, b: torch.Tensor, indices: t.Tuple[t.List[int], t.List[int]] = ([1], [0]), moveaxis=None):
    result = torch.tensordot(a, b, indices)
    if moveaxis is not None:
        result = torch.moveaxis(result, *moveaxis)
    return result

def trace(a):
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


def test_measurement_noise(
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    varying_input_idx: t.Optional[t.List[int]],
    max_variance: float = 1.,
    step: float = 0.1,
) -> t.Dict[str, t.List[float]]:
    
    num_qubits = 2 # tested for 2 qubits
    dim = 2**num_qubits
    
    tomography = Tomography(num_qubits, Kwiat_projectors)
    tomography.calulate_B_inv()
    
    metrics = {}
    for variance in np.arange(0, max_variance, step):
        num_skipped = 0
        metrics[variance] = {name: 0 for name in criterions.keys()}
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f' Variance: {variance}'):
                data_min = torch.maximum(data[:, torch.tensor(varying_input_idx)] - variance, torch.zeros_like(data[:, torch.tensor(varying_input_idx)]))
                data_max = torch.minimum(data[:, torch.tensor(varying_input_idx)] + variance, torch.ones_like(data[:, torch.tensor(varying_input_idx)]))
                interval = data_max - data_min + 1e-6
                varied_data = torch.rand_like(interval) * interval + data_min
                data[:, torch.tensor(varying_input_idx)] = varied_data

                predictions = []
                for measurements in data:
                    rho_rec = tomography.reconstruct(measurements.numpy())
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

                for name, criterion in criterions.items():
                    interval[predictions == -1] = 0
                    metrics[variance][name] += ((criterion(predictions, target) * interval).sum() / interval.sum()).item()
        print(f'Percentage of skipped samples (due to incorrect reconstruction): {num_skipped/len(test_loader.dataset)*100:.2f}')
        for name in metrics[variance].keys():
            metrics[variance][name] /= len(test_loader)
            print(f'{name} - variance {variance}: {metrics[variance][name]:.4f}')
    return metrics

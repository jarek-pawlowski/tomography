from itertools import product
from functools import reduce
import typing as t

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.tomography_utils_torch import measure
from src.tomography_utils_numpy import Kwiat

def collect_rhos_with_closest_kwiat_bases(
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


def collect_kwiat_measurements_basis_probabilities_from_discrete_model(
        model: nn.Module,
        device: torch.device,
        first_measurement: torch.Tensor,
        rho: torch.Tensor,
):
    model.eval()
    model.to(device)

    bases = [
        torch.from_numpy(base).to(device).to(torch.complex64)
        for base in Kwiat.basis
    ]
    qubits_bases = [torch.stack(multi_qubit_base) for multi_qubit_base in product(bases, repeat=model.num_qubits)]
    qubits_bases = torch.stack(qubits_bases)

    rho, first_measurement = rho.to(device), first_measurement.to(device)
    measurement_with_basis = [
        (first_measurement[i:i+1].unsqueeze(0), qubits_bases[i].unsqueeze(0))
        for i in range(first_measurement.shape[0])
    ]
    predicted_best_rhos, predicted_bases_probabilities, predicted_all_rhos = model(measurement_with_basis, rho.unsqueeze(0))
    probabilites = reduce(torch.func.vmap(torch.kron), [predicted_bases_probabilities[0, :, j] for j in range(predicted_bases_probabilities.shape[2])])
    return probabilites, predicted_best_rhos[0]


def collect_measurements_outputs_from_model(
        model: nn.Module,
        device: torch.device,
        first_measurement: torch.Tensor,
        rho: torch.Tensor,
):
    model.eval()
    model.to(device)

    rho, first_measurement = rho.to(device).unsqueeze(0), first_measurement.to(device).unsqueeze(0)
    basis = torch.from_numpy(Kwiat.basis[0]).to(device).to(torch.complex64)
    basis = basis.unsqueeze(0).expand(rho.shape[0], -1, -1)
    measurement_with_basis = (first_measurement[:, 0:1], torch.stack((basis, basis), dim=1))

    predicted_rhos, predicted_bases = model(measurement_with_basis, rho)
    rho_complex = torch.complex(rho[0, 0], rho[0, 1]).view(*[2, 2]*2)
    measurements = torch.stack([measure(rho_complex, (basis_i[0], basis_i[1])) for basis_i in predicted_bases[0]])
    return predicted_bases[0], measurements, predicted_rhos[0]


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

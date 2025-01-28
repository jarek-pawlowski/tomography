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
    mode: str = 'shift_feature',
    features_value_range: t.Tuple[int, int] = (0., 1.),
    noise_variance: float = 0.5,
    num_steps: int = 10,
    specific_feature_idx: t.Optional[int] = None,
    label_separate_output: bool = False,
    split_thresholds: t.List[float] = [0., 1.e-6, 1.01],
) -> float:

    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)

    num_classes = len(split_thresholds) - 1
    if label_separate_output:
        separate_avg_output = {label: 0. for label in range(num_classes)}
        num_samples = {label: 0 for label in range(num_classes)}
    avg_output = 0.
    with torch.no_grad():
        for data, y in tqdm(test_loader, desc=f' Calculating average model output...'):
            data = data.to(device)
            if specific_feature_idx is not None:
                num_features = 1
                if label_separate_output:
                    for label in range(num_classes):
                        data_label = data[torch.logical_and(y.squeeze() >= split_thresholds[label], y.squeeze() < split_thresholds[label + 1])]
                        avg_output_label = calculate_cumulative_model_output_varying_feature(
                            model, data_label, specific_feature_idx, features_value_range, mode, noise_variance, num_steps
                        )
                        separate_avg_output[label] += avg_output_label
                        num_samples[label] += len(data_label)
                        avg_output += avg_output_label
                else:
                    avg_output += calculate_cumulative_model_output_varying_feature(
                        model, data, specific_feature_idx, features_value_range, mode, noise_variance, num_steps
                    )
            else:
                num_features = data.shape[1]
                for feature_idx in range(num_features):
                    if label_separate_output:
                        for label in range(num_classes):
                            data_label = data[torch.logical_and(y.squeeze() >= split_thresholds[label], y.squeeze() < split_thresholds[label + 1])]
                            avg_output_label = calculate_cumulative_model_output_varying_feature(
                                model, data_label, feature_idx, features_value_range, mode, noise_variance, num_steps
                            )
                            separate_avg_output[label] += avg_output_label
                            if feature_idx == 0:
                                num_samples[label] += len(data_label)
                            avg_output += avg_output_label
                    else:
                        avg_output += calculate_cumulative_model_output_varying_feature(
                            model, data, feature_idx, features_value_range, mode, noise_variance, num_steps
                        )
    avg_output /= (len(test_loader.dataset) * num_steps * num_features)
    separate_avg_output = {label: (separate_avg_output[label] / (num_samples[label] * num_steps * num_features)) for label in range(num_classes)}
    # avg_output /= (len(test_loader) * num_steps * num_features)
    if label_separate_output:
        return avg_output, separate_avg_output
    return avg_output


def calculate_cumulative_model_output_varying_feature(
    model: t.Union[nn.Module, t.Callable],
    data: torch.Tensor,
    feature_idx: int,
    features_value_range: t.Tuple[int, int] = (0., 1.),
    mode: str = 'shift_feature',
    noise_variance: float = 0.5,
    num_steps: int = 10,
    feature_aggregate: str = 'sum',
    batch_aggregate: str = 'sum',
):
    cumulative_output = torch.zeros_like(model(data))
    if mode == 'shift_feature':
        step = (features_value_range[1] - features_value_range[0]) / num_steps
        for feature_value in np.arange(*features_value_range, step):
            new_data = data.clone()
            new_data[..., torch.tensor([feature_idx])] = feature_value
            output = model(new_data)
            cumulative_output += output
    elif mode == 'add_noise':
        for i in range(num_steps):
            new_data = data.clone()
            data_min = torch.maximum(new_data[..., torch.tensor(feature_idx)] - noise_variance, torch.zeros_like(new_data[..., torch.tensor(feature_idx)]))
            data_max = torch.minimum(new_data[..., torch.tensor(feature_idx)] + noise_variance, torch.ones_like(new_data[..., torch.tensor(feature_idx)]))
            interval = data_max - data_min + 1e-6
            varied_data = torch.rand_like(interval) * interval + data_min
            new_data[:, torch.tensor(feature_idx)] = varied_data
            output = model(new_data)
            # output = ((model(new_data) * interval).sum() / interval.sum()).item()
            cumulative_output += output
    if feature_aggregate == 'mean':
        cumulative_output /= num_steps
    if batch_aggregate == 'mean':
        cumulative_output = cumulative_output.mean().item()
    if batch_aggregate == 'sum':
        cumulative_output = cumulative_output.sum().item()
    return cumulative_output

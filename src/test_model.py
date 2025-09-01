from collections import defaultdict
from functools import reduce
from itertools import product
import random
import numpy as np
from tqdm import tqdm
import typing as t

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tomography_utils_torch import reconstruct, reconstruct_with_nn_corrections
from src.data_utils import generate_sample_from_mean_and_covariance
from src.tomography_utils_numpy import N_QUBIT_GAMMAS, Kwiat, Kwiat_projectors
from src.train import reconstruct_rho_from_corrections_dynamic_selection
from src.model_utils import calculate_cumulative_model_output_varying_feature


def test(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable] = {},
    binlike_criterions: t.Dict[str, t.Callable] = {},
    num_bins: t.Optional[int] = None
) -> t.Dict[str, t.List[float]]:

    model.eval()
    model.to(device)

    metrics = {name: 0 for name in criterions.keys()}
    binlike_metrics = {name: {i: [] for i in range(num_bins)} for name in binlike_criterions.keys()}

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing model...'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            for name, criterion in criterions.items():
                metrics[name] += criterion(output, target).item()
            for name, criterion in binlike_criterions.items():
                criterion_metrics = criterion(output, target)
                for bin_idx, metric_value in criterion_metrics.items():
                    binlike_metrics[name][bin_idx].append(metric_value)

    for name in metrics.keys():
        metrics[name] /= len(test_loader)
        print(f'{name}: {metrics[name]:.4f}')

    for name in binlike_metrics.keys():
        keys_to_remove = []
        for bin_idx in binlike_metrics[name].keys():
            if len(binlike_metrics[name][bin_idx]) == 0:
                keys_to_remove.append(bin_idx)
                continue
            binlike_metrics[name][bin_idx] = torch.mean(torch.stack(binlike_metrics[name][bin_idx])).item()
            print(f'{name} - bin {bin_idx}: {binlike_metrics[name][bin_idx]:.4f}')
        for key in keys_to_remove:
            binlike_metrics[name].pop(key)
    
    return {**metrics, **binlike_metrics}



def test_measurement_predictor(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    max_num_measurements: int = 16,
    mode: str = 'rho' # 'rho' or 'concurrence'
) -> t.Dict[str, t.List[float]]:

    model.eval()
    model.to(device)

    metrics = {name: {f'measurement {i}': 0 for i in range(max_num_measurements)} for name in criterions.keys()}
    with torch.no_grad():
        for rho, measurement, concurrence in tqdm(test_loader, desc='Testing model...'):
            rho, measurement = rho.to(device), measurement.to(device)
            if mode == 'rho':
                target = rho.to(device)
            elif mode == 'concurrence':
                target = concurrence.to(device)
            else:
                raise ValueError(f'Unknown mode: {mode}')
            basis = torch.from_numpy(Kwiat.basis[0]).to(device).to(torch.complex64)
            basis = basis.unsqueeze(0).expand(rho.shape[0], -1, -1)
            measurement_with_basis = (measurement[:, 0:1], torch.stack([basis]*model.num_qubits, dim=1))
            predicted_target, _ = model(measurement_with_basis, rho)
            for name, criterion in criterions.items():
                for i in range(predicted_target.shape[1]):
                    metrics[name][f'measurement {i}'] += criterion(predicted_target[:, i], target).item()
    for name in metrics.keys():
        for i in range(max_num_measurements):
            metrics[name][f'measurement {i}'] /= len(test_loader)
            print(f'{name} - measurement {i}: {metrics[name][f"measurement {i}"]:.4f}')
    return metrics


def test_reconstructor(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    measurements_subset: t.Optional[t.Union[int, t.List[int]]] = None
) -> t.Dict[str, t.List[float]]:

    model.eval()
    model.to(device)
    metrics = {name: 0 for name in criterions.keys()}

    single_qubits_basis_matrices = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat.basis]
    two_qubits_basis_matrices = torch.stack([torch.stack([basis1, basis2]) for basis1, basis2 in product(single_qubits_basis_matrices, repeat=2)])
    selected_basis_matrices = two_qubits_basis_matrices
    num_qubits = 2

    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc='Testing model...'):
            rho, measurement = rho.to(device), measurement.to(device)
            if type(measurements_subset) == int:
                measurements_subset = random.sample(range(measurement.shape[1]), measurements_subset)
            if measurements_subset is not None:
                measurement = measurement[:, measurements_subset]
                selected_basis_matrices = two_qubits_basis_matrices[measurements_subset]

            selected_basis_matrices = selected_basis_matrices.unsqueeze(0).expand(rho.shape[0], -1, -1, -1, -1) # expand for batch dimension
            basis_as_vector = torch.stack((selected_basis_matrices.real, selected_basis_matrices.imag), dim=-1).view(-1, selected_basis_matrices.shape[1]*num_qubits*2*2*2)
            measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
            predicted_rhos = model(measurement_predictor_input)

            for name, criterion in criterions.items():
                metrics[name] += criterion(predicted_rhos, rho).item()
    for name in metrics.keys():
        metrics[name] /= len(test_loader)
        print(f'{name}: {metrics[name]:.4f}')
    return metrics


def test_gammas_reconstructor(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    measurements_subset: t.Optional[t.Union[int, t.List[int]]] = None,
    model_input_info: str = 'full' # 'full', 'measurement' or 'measurement_basis'
) -> t.Dict[str, t.List[float]]:

    model.eval()
    model.to(device)
    metrics = {name: 0 for name in criterions.keys()}

    single_qubits_basis_matrices = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat.basis]
    two_qubits_basis_matrices = torch.stack([torch.stack([basis1, basis2]) for basis1, basis2 in product(single_qubits_basis_matrices, repeat=2)])
    selected_basis_matrices = two_qubits_basis_matrices
    single_qubits_projection_vectors = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat_projectors.basis]
    two_qubits_projection_vectors = torch.stack([torch.kron(basis1, basis2) for basis1, basis2 in product(single_qubits_projection_vectors, repeat=2)])
    selected_projection_vectors = two_qubits_projection_vectors
    num_qubits = 2

    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc='Testing model...'):
            rho, measurement = rho.to(device), measurement.to(device)
            if type(measurements_subset) == int:
                measurements_subset = random.sample(range(measurement.shape[1]), measurements_subset)
            if measurements_subset is not None:
                measurement = measurement[:, measurements_subset]
                selected_basis_matrices = two_qubits_basis_matrices[measurements_subset]
                selected_projection_vectors = two_qubits_projection_vectors[measurements_subset]

            selected_basis_matrices = selected_basis_matrices.unsqueeze(0).expand(rho.shape[0], -1, -1, -1, -1) # expand for batch dimension
            basis_as_vector = torch.stack((selected_basis_matrices.real, selected_basis_matrices.imag), dim=-1).view(-1, selected_basis_matrices.shape[1]*num_qubits*2*2*2)
            if model_input_info == 'full':
                measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
            elif model_input_info == 'measurement':
                measurement_predictor_input = measurement
            elif model_input_info == 'measurement_basis':
                measurement_predictor_input = basis_as_vector
            else:
                raise ValueError(f'Unknown model_input_info: {model_input_info}')
            predicted_gammas = model(measurement_predictor_input)
            complex_gammas = torch.complex(predicted_gammas[:, :, 0], predicted_gammas[:, :, 1])
            reconstructed_rho = torch.stack([reconstruct(measurement_i, selected_projection_vectors, gammas_i) for measurement_i, gammas_i in zip(measurement, complex_gammas)])
            reconstructed_rho = torch.stack([reconstructed_rho.real, reconstructed_rho.imag], dim=1)

            for name, criterion in criterions.items():
                metrics[name] += criterion(reconstructed_rho, rho).item()
    for name in metrics.keys():
        metrics[name] /= len(test_loader)
        print(f'{name}: {metrics[name]:.4f}')
    return metrics


def test_tomography_corrections_predictor(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    measurements_subset: t.Optional[t.Union[int, t.List[int]]] = None,
    model_input_info: str = 'full', # 'full', 'measurement' or 'measurement_basis'
    std_out: t.Optional[t.IO] = None,
    trace_normalization: bool = False,
    use_m2_corrections: bool = False,
) -> t.Dict[str, t.List[float]]:

    model.eval()
    model.to(device)
    metrics = {name: 0 for name in criterions.keys()}

    num_qubits = test_loader.dataset.num_qubits

    single_qubits_basis_matrices = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat.basis]
    n_qubits_basis_matrices = torch.stack([torch.stack(multi_qubit_base) for multi_qubit_base in product(single_qubits_basis_matrices, repeat=num_qubits)])
    # two_qubits_basis_matrices = torch.stack([torch.stack([basis1, basis2]) for basis1, basis2 in product(single_qubits_basis_matrices, repeat=2)])
    selected_basis_matrices = n_qubits_basis_matrices

    single_qubits_projection_vectors = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat_projectors.basis]
    n_qubits_projection_vectors = torch.stack([reduce(torch.kron, [basis_i for basis_i in basis]) for basis in product(single_qubits_projection_vectors, repeat=num_qubits)])
    # two_qubits_projection_vectors = torch.stack([torch.kron(basis1, basis2) for basis1, basis2 in product(single_qubits_projection_vectors, repeat=2)])
    selected_projection_vectors = n_qubits_projection_vectors

    # gammas = torch.tensor(Gammas, dtype=torch.complex64, device=device)
    gammas = torch.tensor(N_QUBIT_GAMMAS(num_qubits), dtype=torch.complex64, device=device)

    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc='Testing model...', file=std_out):
            rho, measurement = rho.to(device), measurement.to(device)
            if type(measurements_subset) == int:
                measurements_subset = random.sample(range(measurement.shape[1]), measurements_subset)
            if measurements_subset is not None:
                measurement = measurement[:, measurements_subset]
                selected_basis_matrices = n_qubits_basis_matrices[measurements_subset]
                selected_projection_vectors = n_qubits_projection_vectors[measurements_subset]

            selected_basis_matrices = selected_basis_matrices.unsqueeze(0).expand(rho.shape[0], -1, -1, -1, -1) # expand for batch dimension
            basis_as_vector = torch.stack((selected_basis_matrices.real, selected_basis_matrices.imag), dim=-1).view(-1, selected_basis_matrices.shape[1]*num_qubits*2*2*2)
            if model_input_info == 'full':
                measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
            elif model_input_info == 'measurement':
                measurement_predictor_input = measurement
            elif model_input_info == 'measurement_basis':
                measurement_predictor_input = basis_as_vector
            else:
                raise ValueError(f'Unknown model_input_info: {model_input_info}')

            if use_m2_corrections:
                m2_corrections, inverse_corrections, r_corrections = model(measurement_predictor_input)
                complex_m2_corrections = torch.complex(m2_corrections[:, 0], m2_corrections[:, 1])
            else:
                inverse_corrections, r_corrections = model(measurement_predictor_input)
                complex_m2_corrections = None

            complex_inverse_corrections = torch.complex(inverse_corrections[:, 0], inverse_corrections[:, 1])
            complex_r_corrections = torch.complex(r_corrections[:, 0], r_corrections[:, 1])
            reconstructed_rho = torch.stack([
                reconstruct_with_nn_corrections(
                    measurement_i,
                    selected_projection_vectors,
                    gammas,
                    inverse_correction_i,
                    r_correction_i,
                    complex_m2_corrections[i] if complex_m2_corrections is not None else None
                )
                for i, (measurement_i, inverse_correction_i, r_correction_i) in enumerate(zip(measurement, complex_inverse_corrections, complex_r_corrections))
            ])
            if trace_normalization:
                reconstructed_rho = reconstructed_rho / torch.vmap(torch.trace)(reconstructed_rho).view(-1, 1, 1)
            reconstructed_rho = torch.stack([reconstructed_rho.real, reconstructed_rho.imag], dim=1)

            for name, criterion in criterions.items():
                metrics[name] += criterion(reconstructed_rho, rho)
    for name in metrics.keys():
        metrics[name] /= len(test_loader)
        try:
            metrics[name] = metrics[name].item()
            print(f'{name}: {metrics[name]:.4f}', file=std_out)
        except:
            pass
    return metrics


def test_measurement_projector_predictor(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    max_num_measurements: int = 16,
) -> t.Dict[str, t.List[float]]:
    
    model.eval()
    model.to(device)

    num_qubits = test_loader.dataset.num_qubits
    gammas = torch.tensor(N_QUBIT_GAMMAS(num_qubits), dtype=torch.complex64, device=device)
    
    metrics = {name: {f'measurement {i}': 0 for i in range(max_num_measurements)} for name in criterions.keys()}
    with torch.no_grad():
        for rho, measurement, concurrence in tqdm(test_loader, desc='Testing model...'):
            rho, measurement = rho.to(device), measurement.to(device)
            target = rho.to(device)

            projector = torch.from_numpy(Kwiat_projectors.basis[0]).to(device).to(torch.complex64).squeeze(-1)
            projector = projector.unsqueeze(0).expand(rho.shape[0], -1)
            measurement_with_projector = (measurement[:, 0:1], torch.stack([projector]*model.num_qubits, dim=1))
            predicted_projectors, predicted_measurements = model(measurement_with_projector, rho)

            n_qubits_projection_vectors = torch.stack([
                reduce(torch.vmap(torch.kron), [predicted_projectors[:, i, j] for j in range(predicted_projectors.shape[2])])
                for i in range(predicted_projectors.shape[1])
            ], dim=1)

            batch_target = []
            for k in range(n_qubits_projection_vectors.shape[0]):
                predicted_target = torch.stack(
                    [
                        reconstruct(
                            predicted_measurements[k, :i],
                            n_qubits_projection_vectors[k, :i].unsqueeze(-1),
                            gammas,
                            enforce_valid_density_matrix=False,
                            inverse="pinv"
                        )
                        for i in range(1, n_qubits_projection_vectors.shape[1] + 1)
                    ],
                    dim=0
                )
                predicted_target = torch.stack([predicted_target.real, predicted_target.imag], dim=1)
                batch_target.append(predicted_target)
            predicted_target = torch.stack(batch_target)

            for name, criterion in criterions.items():
                for i in range(predicted_target.shape[1]):
                    metrics[name][f'measurement {i}'] += criterion(predicted_target[:, i], target).item()
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
    mode: str = 'rho' # 'rho' or 'concurrence'
) -> t.Dict[str, t.List[float]]:

    model.eval()
    model.to(device)

    bases = [
        torch.from_numpy(base).to(device).to(torch.complex64)
        for base in Kwiat.basis
    ]
    qubits_bases = [torch.stack(multi_qubit_base) for multi_qubit_base in product(bases, repeat=model.num_qubits)]
    qubits_bases = torch.stack(qubits_bases)

    single_qubits_projection_vectors = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat_projectors.basis]
    n_qubits_projection_vectors = torch.stack([reduce(torch.kron, [basis_i for basis_i in basis]) for basis in product(single_qubits_projection_vectors, repeat=model.num_qubits)])
    gammas = torch.tensor(N_QUBIT_GAMMAS(model.num_qubits), dtype=torch.complex64, device=device)

    metrics = {name: {f'measurement {i}': 0 for i in range(max_num_measurements)} for name in criterions.keys()}
    with torch.no_grad():
        for rho, measurement, concurrence in tqdm(test_loader, desc='Testing model...'):
            rho, measurement = rho.to(device), measurement.to(device)
            qubits_bases_batch = qubits_bases.unsqueeze(0).expand(rho.shape[0], -1, -1, -1, -1)
            measurement_with_basis = [
                (measurement[:, i:i+1], qubits_bases_batch[:, i])
                for i in range(measurement.shape[1])
            ]
            if mode == 'rho' or mode == 'tomo_corrections':
                target = rho.to(device)
            elif mode == 'concurrence':
                target = concurrence.to(device)
            else:
                raise ValueError(f'Unknown mode: {mode}')

            predicted_best_targets, predicted_bases_probabilities, _ = model(measurement_with_basis, rho)
            for name, criterion in criterions.items():
                for i in range(predicted_best_targets.shape[1]):
                    if mode == 'tomo_corrections':
                        probabilites = reduce(torch.func.vmap(torch.kron), [predicted_bases_probabilities[:, i, j] for j in range(predicted_bases_probabilities.shape[2])])
                        selected_projection_vectors_ids = torch.argsort(probabilites, dim=1, descending=True)[:, :i+1]
                        selected_projection_vectors = n_qubits_projection_vectors[selected_projection_vectors_ids]
                        selected_measurements = torch.stack([m[selected_projection_vectors_ids[i]] for i, m in enumerate(measurement)])
                        inverse_corrections, r_corrections = predicted_best_targets[:, :i+1, :, 0], predicted_best_targets[:, i, :, 1]
                        inverse_corrections = torch.moveaxis(inverse_corrections, 1, -2)                        
                        complex_inverse_corrections = torch.complex(inverse_corrections[:, 0], inverse_corrections[:, 1])
                        complex_r_corrections = torch.complex(r_corrections[:, 0], r_corrections[:, 1])
                        reconstructed_rho = reconstruct_rho_from_corrections_dynamic_selection(selected_projection_vectors, gammas, selected_measurements, complex_inverse_corrections, complex_r_corrections)
                        metrics[name][f'measurement {i}'] += criterion(reconstructed_rho, target).item()
                    else:
                        metrics[name][f'measurement {i}'] += criterion(predicted_best_targets[:, i], target).item()
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


def test_varying_feature_with_value_range(
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


def test_varying_feature_with_noise(
    model: t.Union[nn.Module, t.Callable],
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    feature_idx: t.Optional[t.List[int]],
    feature_noise_variance: float = 0.5,
    num_repetitions: int = 10,
    model_output_mean: t.Optional[t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]] = None,
    label_separate_output: bool = False,
    split_thresholds: t.List[float] = [0., 1.e-6, 1.01],
) -> t.Dict[str, t.List[float]]:

    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)

    avg_outputs = defaultdict(float)
    avg_distances = defaultdict(float)
    num_classes = len(split_thresholds) - 1
    if label_separate_output:
        separate_avg_outputs = {label: defaultdict(float) for label in range(num_classes)}
        separate_avg_distances = {label: defaultdict(float) for label in range(num_classes)}
    for i in range(num_repetitions):
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                data_min = torch.maximum(data[:, torch.tensor(feature_idx)] - feature_noise_variance, torch.zeros_like(data[:, torch.tensor(feature_idx)]))
                data_max = torch.minimum(data[:, torch.tensor(feature_idx)] + feature_noise_variance, torch.ones_like(data[:, torch.tensor(feature_idx)]))
                interval = data_max - data_min + 1e-6
                varied_data = torch.rand_like(interval) * interval + data_min
                data[:, torch.tensor(feature_idx)] = varied_data
                output = model(data)
                if model_output_mean is not None:
                    if label_separate_output:
                        for label in range(num_classes):
                            label_output = output[torch.logical_and(target >= split_thresholds[label], target < split_thresholds[label + 1])]
                            label_model_output_mean = model_output_mean[1][label]
                            separate_avg_distances[label][i] += torch.abs(label_output - label_model_output_mean).mean().item()
                        avg_distances[i] += torch.abs(output - model_output_mean[0]).mean().item()
                    else:
                    # avg_distances[i] += ((torch.abs(output - model_output_mean) * interval).sum() / interval.sum()).item()
                        avg_distances[i] += torch.abs(output - model_output_mean).mean().item()
                # avg_outputs[i] += ((output * interval).sum() / interval.sum()).item()    
                if label_separate_output:
                    for label in range(num_classes):
                        label_output = output[torch.logical_and(target >= split_thresholds[label], target < split_thresholds[label + 1])]
                        separate_avg_outputs[label][i] += label_output.mean().item()
                avg_outputs[i] += output.mean().item()

        if label_separate_output:
            for label in range(num_classes):
                separate_avg_outputs[label][i] /= len(test_loader)
                separate_avg_distances[label][i] /= len(test_loader)
        avg_outputs[i] /= len(test_loader)
        avg_distances[i] /= len(test_loader)

    outputs = torch.tensor(list(avg_outputs.values()))
    mean_metrics = {
        criterion_name: criterion(outputs) for criterion_name, criterion in criterions.items()
    }
    distances = torch.tensor(list(avg_distances.values()))
    distance_metrics = {
        criterion_name: criterion(distances) for criterion_name, criterion in criterions.items()
    }

    if label_separate_output:
        separate_outputs = {label: torch.tensor(list(separate_avg_outputs[label].values())) for label in range(num_classes)}
        separate_mean_metrics = {
            label: {criterion_name: criterion(separate_outputs[label]) for criterion_name, criterion in criterions.items()}
            for label in range(num_classes)
        }
        separate_distances = {label: torch.tensor(list(separate_avg_distances[label].values())) for label in range(num_classes)}
        separate_distance_metrics = {
            label: {criterion_name: criterion(separate_distances[label]) for criterion_name, criterion in criterions.items()}
            for label in range(num_classes)
        }
        return mean_metrics, distance_metrics, separate_mean_metrics, separate_distance_metrics
    
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


def calculate_model_mad_varying_feature(
    model: t.Union[nn.Module, t.Callable],
    device: torch.device,
    data_loader: DataLoader,
    feature_idx: int,
    feature_noise_variance: float = 0.5,
    num_repetitions: int = 10,
    substract_mean: float = 0.,
):
    if isinstance(model, nn.Module):
        model.eval()
        model.to(device)

    total_output = 0.
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc=f' Calculating average model output with substract mean {substract_mean:.2f}...'):
            data = data.to(device)
            randomized_outputs = torch.stack([
                _calculate_model_output_from_randomized_input(data, model, feature_idx, feature_noise_variance)
                for _ in range(num_repetitions)
            ])

            mad = torch.abs(randomized_outputs - randomized_outputs.mean(dim=0)).mean(dim=0)
            total_output += torch.abs(mad - substract_mean).sum().item()
    total_output /= len(data_loader.dataset)
    print(f'MAD: {total_output:.4f}')
    return total_output


def _calculate_model_output_from_randomized_input(
    data: torch.Tensor,
    model: nn.Module,
    feature_idx: t.List[int],
    feature_noise_variance: float = 0.5,
):
    data_min = torch.maximum(data[:, torch.tensor(feature_idx)] - feature_noise_variance, torch.zeros_like(data[:, torch.tensor(feature_idx)]))
    data_max = torch.minimum(data[:, torch.tensor(feature_idx)] + feature_noise_variance, torch.ones_like(data[:, torch.tensor(feature_idx)]))
    interval = data_max - data_min + 1e-6
    data_i = data.clone()
    varied_data = torch.rand_like(interval) * interval + data_min
    data_i[:, torch.tensor(feature_idx)] = varied_data
    output = model(data_i)
    return output



def test_lstm_reconstructor(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterions: t.Dict[str, t.Callable],
    measurements_order: torch.Tensor,
    std_out: t.Optional[t.IO] = None,
) -> t.Dict[str, float]:


    model.eval()
    model.to(device)

    bases = [
        torch.from_numpy(base).to(device).to(torch.complex64)
        for base in Kwiat.basis
    ]
    qubits_bases = [torch.stack(multi_qubit_base) for multi_qubit_base in product(bases, repeat=model.num_qubits)]
    qubits_bases = torch.stack(qubits_bases)

    max_num_measurements = 4**model.num_qubits

    metrics = {name: {f'measurement {i}': 0 for i in range(max_num_measurements)} for name in criterions.keys()}
    with torch.no_grad():
        for rho, measurement, _ in tqdm(test_loader, desc='Testing model...', file=std_out):
            rho, measurement = rho.to(device), measurement.to(device)
            qubits_bases_batch = qubits_bases.unsqueeze(0).expand(rho.shape[0], -1, -1, -1, -1)

            target = rho.to(device)

            measurements_order_ = measurements_order.to(device).unsqueeze(0).expand(target.shape[0], -1)

            measurements_with_basis = []
            for i in range(measurement.shape[1]):
                measurement_i = measurement[:, i:i+1]
                basis_i = qubits_bases_batch[:, i]
                basis_as_vector = torch.stack((basis_i.real, basis_i.imag), dim=-1).view(-1, model.num_qubits*2*2*2)
                reconstructor_input = torch.cat((measurement_i, basis_as_vector), dim=-1)
                measurements_with_basis.append(reconstructor_input)

            measurements_with_basis = torch.stack(measurements_with_basis, dim=1) # shape (batch, max_num_measurements, num_qubits*2*2*2 + 1)
            measurements_with_basis_ordered = torch.gather(measurements_with_basis, 1, measurements_order_.unsqueeze(-1).expand(-1, -1, measurements_with_basis.shape[-1]))  # shape (batch, max_num_measurements, num_qubits*2*2*2 + 1)

            predictions  = model(measurements_with_basis_ordered)
            for name, criterion in criterions.items():
                for i in range(predictions.shape[1]):
                    metrics[name][f'measurement {i}'] += criterion(predictions[:, i], target).item()

    for name in metrics.keys():
        for i in range(max_num_measurements):
            metrics[name][f'measurement {i}'] /= len(test_loader)
            print(f'{name} - measurement {i}: {metrics[name][f"measurement {i}"]:.4f}')
    return metrics

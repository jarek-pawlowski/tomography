from functools import reduce
from itertools import product
import random
import typing as t

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.tomography_utils_torch import calculate_B, reconstruct, reconstruct_with_nn_corrections, reconstruct_with_nn_corrections_and_B_inv
from src.tomography_utils_numpy import N_QUBIT_GAMMAS, Kwiat, Kwiat_projectors


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
    bases_loss_fn: t.Optional[t.Callable] = None,
    bases_loss_weight: float = 1,
    mode: str = 'rho', # 'rho' or 'concurrence'
    increase_loss_weights_with_measurement: bool = False,
    add_noise_to_measurement_basis: bool = False,
    contrastive_loss_start_epoch: int = 0,
) -> t.Dict[str, t.List[float]]:

    model.train()
    model.to(device)
    metrics = {'train_loss': 0, 'bases_loss': 0}
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (rho, measurement, concurrence) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        if mode == 'rho':
            target = rho.to(device)
        elif mode == 'concurrence':
            target = concurrence.to(device)
        else:
            raise ValueError(f'Unknown mode: {mode}')
        optimizer.zero_grad()
        basis = torch.from_numpy(Kwiat.basis[0]).to(device).to(torch.complex64)
        basis = basis.unsqueeze(0).expand(rho.shape[0], -1, -1)
        measurement_with_basis = (measurement[:, 0:1], torch.stack([basis]*model.num_qubits, dim=1))
        if add_noise_to_measurement_basis:
            predicted_target, predicted_bases = model(measurement_with_basis, rho, add_noise_to_measurement_basis=True)
        else:
            predicted_target, predicted_bases = model(measurement_with_basis, rho)

        loss = torch.zeros(1).to(device)
        for i in range(predicted_target.shape[1]):
            loss_weight_i = 1
            if increase_loss_weights_with_measurement:
                    loss_weight_i = (i + 1) # / (predicted_target.shape[1] * (predicted_target.shape[1] + 1) / 2)
            if epoch < 10 and mode == 'concurrence':
                predicted_target_with_noise = predicted_target[:, i] + torch.randn_like(predicted_target[:, i]) * 1e-3
                loss += criterion(predicted_target_with_noise, target) * loss_weight_i
            else:
                loss += criterion(predicted_target[:, i], target) * loss_weight_i
        if (bases_loss_fn is not None) and (epoch > contrastive_loss_start_epoch):
            bases_loss = bases_loss_fn(predicted_bases)
            metrics['bases_loss'] += bases_loss.item()
            loss += bases_loss_weight * bases_loss
        loss.backward()
        optimizer.step()
        metrics['train_loss'] += loss.item()
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    metrics['train_loss'] /= len(train_loader)
    metrics['bases_loss'] /= len(train_loader)
    return metrics


def train_reconstructor(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    log_interval: int = 100,
    criterion: t.Callable = nn.MSELoss(),
    measurements_subset: t.Optional[t.Union[int, t.List[int]]] = None
) -> t.Dict[str, t.List[float]]:

    model.train()
    model.to(device)
    metrics = {'train_loss': 0}
    single_qubits_basis_matrices = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat.basis]
    two_qubits_basis_matrices = torch.stack([torch.stack([basis1, basis2]) for basis1, basis2 in product(single_qubits_basis_matrices, repeat=2)])
    selected_basis_matrices = two_qubits_basis_matrices
    num_qubits = 2

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (rho, measurement, _) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        optimizer.zero_grad()
        if type(measurements_subset) == int:
            measurements_subset = random.sample(range(measurement.shape[1]), measurements_subset)
        if measurements_subset is not None:
            measurement = measurement[:, measurements_subset]
            selected_basis_matrices = two_qubits_basis_matrices[measurements_subset]

        selected_basis_matrices = selected_basis_matrices.unsqueeze(0).expand(rho.shape[0], -1, -1, -1, -1) # expand for batch dimension
        basis_as_vector = torch.stack((selected_basis_matrices.real, selected_basis_matrices.imag), dim=-1).view(-1, selected_basis_matrices.shape[1]*num_qubits*2*2*2)
        measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
        predicted_rhos = model(measurement_predictor_input)
        loss = criterion(predicted_rhos, rho)
        loss.backward()
        optimizer.step()
        metrics['train_loss'] += loss.item()
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    metrics['train_loss'] /= len(train_loader)
    return metrics


def train_gammas_reconstructor(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    log_interval: int = 100,
    criterion: t.Callable = nn.MSELoss(),
    measurements_subset: t.Optional[t.Union[int, t.List[int]]] = None,
    model_input_info: str = 'full' # 'full', 'measurement' or 'measurement_basis'
) -> t.Dict[str, t.List[float]]:

    model.train()
    model.to(device)
    metrics = {'train_loss': 0}
    single_qubits_basis_matrices = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat.basis]
    two_qubits_basis_matrices = torch.stack([torch.stack([basis1, basis2]) for basis1, basis2 in product(single_qubits_basis_matrices, repeat=2)])
    selected_basis_matrices = two_qubits_basis_matrices
    single_qubits_projection_vectors = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat_projectors.basis]
    two_qubits_projection_vectors = torch.stack([torch.kron(basis1, basis2) for basis1, basis2 in product(single_qubits_projection_vectors, repeat=2)])
    selected_projection_vectors = two_qubits_projection_vectors
    num_qubits = 2

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (rho, measurement, _) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        optimizer.zero_grad()
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
        loss = criterion(reconstructed_rho, rho)
        loss.backward()
        optimizer.step()
        metrics['train_loss'] += loss.item()
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    metrics['train_loss'] /= len(train_loader)
    return metrics


def train_tomography_corrections_predictor(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    log_interval: int = 100,
    criterion: t.Callable = nn.MSELoss(),
    measurements_subset: t.Optional[t.Union[int, t.List[int]]] = None,
    model_input_info: str = 'full', # 'full', 'measurement' or 'measurement_basis'
    std_out: t.Optional[t.IO] = None
) -> t.Dict[str, t.List[float]]:

    model.train()
    model.to(device)
    metrics = {'train_loss': 0}
    num_qubits = train_loader.dataset.num_qubits

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

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}', file=std_out)
    for batch_idx, (rho, measurement, _) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        optimizer.zero_grad()
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
        inverse_corrections, r_corrections = model(measurement_predictor_input)
        loss = corrections_loss(criterion, selected_projection_vectors, gammas, rho, measurement, inverse_corrections, r_corrections)
        loss.backward()
        optimizer.step()
        metrics['train_loss'] += loss.item()
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': loss.item()})
    metrics['train_loss'] /= len(train_loader)
    return metrics


def corrections_loss(
    criterion: t.Callable,
    selected_projection_vectors: torch.Tensor,
    gammas: torch.Tensor,
    rho: torch.Tensor,
    measurement: torch.Tensor,
    inverse_corrections: torch.Tensor,
    r_corrections: torch.Tensor,
):
    complex_inverse_corrections = torch.complex(inverse_corrections[:, 0], inverse_corrections[:, 1])
    complex_r_corrections = torch.complex(r_corrections[:, 0], r_corrections[:, 1])
    reconstructed_rho = reconstruct_rho_from_corrections(selected_projection_vectors, gammas, measurement, complex_inverse_corrections, complex_r_corrections)
    reconstruction_loss = criterion(reconstructed_rho, rho)
    B = calculate_B(selected_projection_vectors, gammas).to(rho.device)
    total_corrections = torch.stack([
        torch.matmul(B, (torch.matmul(inverse_correction_i.T, measurement_i.to(torch.complex64)) + r_correction_i))
        for measurement_i, inverse_correction_i, r_correction_i in zip(measurement, complex_inverse_corrections, complex_r_corrections)
    ])
    total_corrections = torch.stack([total_corrections.real, total_corrections.imag], dim=1)
    corrections_regularization_loss = torch.nn.functional.mse_loss(total_corrections, torch.zeros_like(total_corrections))
    loss = reconstruction_loss + 0.1*corrections_regularization_loss
    return loss


def corrections_loss_with_dynamic_selection(
    criterion: t.Callable,
    selected_projection_vectors: torch.Tensor,
    gammas: torch.Tensor,
    rho: torch.Tensor,
    measurement: torch.Tensor,
    inverse_corrections: torch.Tensor,
    r_corrections: torch.Tensor,
):
    complex_inverse_corrections = torch.complex(inverse_corrections[:, 0], inverse_corrections[:, 1])
    complex_r_corrections = torch.complex(r_corrections[:, 0], r_corrections[:, 1])
    reconstructed_rho = reconstruct_rho_from_corrections_dynamic_selection(selected_projection_vectors, gammas, measurement, complex_inverse_corrections, complex_r_corrections)
    reconstruction_loss = criterion(reconstructed_rho, rho)

    B = torch.stack([calculate_B(selected_projection_vectors_i, gammas).to(rho.device) for selected_projection_vectors_i in selected_projection_vectors])
    total_corrections = torch.stack([
        torch.matmul(B_i, (torch.matmul(inverse_correction_i.T, measurement_i.to(torch.complex64)) + r_correction_i))
        for B_i, measurement_i, inverse_correction_i, r_correction_i in zip(B, measurement, complex_inverse_corrections, complex_r_corrections)
    ])

    total_corrections = torch.stack([total_corrections.real, total_corrections.imag], dim=1)
    corrections_regularization_loss = torch.nn.functional.mse_loss(total_corrections, torch.zeros_like(total_corrections))
    loss = reconstruction_loss + 0.1*corrections_regularization_loss
    return loss


def reconstruct_rho_from_corrections(selected_projection_vectors: torch.Tensor, gammas: torch.Tensor, measurement: torch.Tensor, complex_inverse_corrections: torch.Tensor, complex_r_corrections: torch.Tensor):
    reconstructed_rho = torch.stack([
            reconstruct_with_nn_corrections(measurement_i, selected_projection_vectors, gammas, inverse_correction_i, r_correction_i)
            for measurement_i, inverse_correction_i, r_correction_i in zip(measurement, complex_inverse_corrections, complex_r_corrections)
        ])
    reconstructed_rho = torch.stack([reconstructed_rho.real, reconstructed_rho.imag], dim=1)
    return reconstructed_rho


def reconstruct_rho_from_corrections_dynamic_selection(selected_projection_vectors: torch.Tensor, gammas: torch.Tensor, measurement: torch.Tensor, complex_inverse_corrections: torch.Tensor, complex_r_corrections: torch.Tensor):
    reconstructed_rho = torch.stack([
            reconstruct_with_nn_corrections(measurement_i, selected_projection_vectors_i, gammas, inverse_correction_i, r_correction_i)
            for measurement_i, selected_projection_vectors_i, inverse_correction_i, r_correction_i in zip(measurement, selected_projection_vectors, complex_inverse_corrections, complex_r_corrections)
        ])
    reconstructed_rho = torch.stack([reconstructed_rho.real, reconstructed_rho.imag], dim=1)
    return reconstructed_rho

def reconstruct_rho_from_corrections_and_B_inv(B_inv: torch.Tensor, gammas: torch.Tensor, measurement: torch.Tensor, complex_inverse_corrections: torch.Tensor, complex_r_corrections: torch.Tensor):
    reconstructed_rho = torch.stack([
            reconstruct_with_nn_corrections_and_B_inv(measurement_i, B_inv, gammas, inverse_correction_i, r_correction_i)
            for measurement_i, inverse_correction_i, r_correction_i in zip(measurement, complex_inverse_corrections, complex_r_corrections)
        ])
    reconstructed_rho = torch.stack([reconstructed_rho.real, reconstructed_rho.imag], dim=1)
    return reconstructed_rho


def train_discrete_measurement_selector(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    reconstructor_optimizer: Optimizer,
    selector_optimizer: Optimizer,
    epoch: int,
    log_interval: int = 100,
    reconstructor_criterion: t.Callable = nn.MSELoss(),
    selector_criterion: t.Callable = nn.CrossEntropyLoss(),
    num_reconstructor_repeats: int = 1,
    num_selector_repeats: int = 1,
    mode: str = 'rho', # 'rho' or 'concurrence'
    num_noisy_epochs: int = 10
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

    single_qubits_projection_vectors = [torch.tensor(basis, dtype=torch.complex64, device=device) for basis in Kwiat_projectors.basis]
    n_qubits_projection_vectors = torch.stack([reduce(torch.kron, [basis_i for basis_i in basis]) for basis in product(single_qubits_projection_vectors, repeat=model.num_qubits)])
    gammas = torch.tensor(N_QUBIT_GAMMAS(model.num_qubits), dtype=torch.complex64, device=device)

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (rho, measurement, concurrence) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        qubits_bases_batch = qubits_bases.unsqueeze(0).expand(rho.shape[0], -1, -1, -1, -1)

        if mode == 'rho' or mode == 'tomo_corrections':
            target = rho.to(device)
        elif mode == 'concurrence':
            target = concurrence.to(device)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        measurement_with_basis = [
            (measurement[:, i:i+1], qubits_bases_batch[:, i])
            for i in range(measurement.shape[1])
        ]

        for _ in range(num_selector_repeats):
            selector_optimizer.zero_grad()
            _, predicted_bases_probabilities, predicted_all_targets = model(measurement_with_basis, rho)
            selector_loss = torch.zeros(1).to(device)
            for i in range(1, predicted_bases_probabilities.shape[1]):
                probabilites = reduce(torch.func.vmap(torch.kron), [predicted_bases_probabilities[:, i, j] for j in range(predicted_bases_probabilities.shape[2])])
                if mode == 'tomo_corrections':
                    selected_projection_vectors_ids = torch.argsort(probabilites, dim=1, descending=True)[:, :i+1]
                    selected_projection_vectors = n_qubits_projection_vectors[selected_projection_vectors_ids]
                    selected_measurements = torch.stack([m[selected_projection_vectors_ids[i]] for i, m in enumerate(measurement)])
                    inverse_corrections, r_corrections = predicted_all_targets[:, :i+1, 1:, :, 0], predicted_all_targets[:, i, 1:, :, 1] # skipping the first basis as it is always chosen as the first measurement
                    inverse_corrections = torch.moveaxis(inverse_corrections, 1, -2)
                    complex_inverse_corrections = torch.complex(inverse_corrections[:, :, 0], inverse_corrections[:, :, 1])
                    complex_r_corrections = torch.complex(r_corrections[:, :, 0], r_corrections[:, :, 1])
                    reconstructed_rhos = torch.stack([
                        reconstruct_rho_from_corrections_dynamic_selection(selected_projection_vectors, gammas, selected_measurements, complex_inverse_corrections[:, k], complex_r_corrections[:, k])
                        for k in range(complex_inverse_corrections.shape[1])
                    ], dim=1).to(device)
                    target_broadcasted = target.unsqueeze(1).expand(-1, predicted_all_targets.shape[2] - 1, *target.shape[1:])
                    reconstruction_losses = torch.nn.functional.mse_loss(reconstructed_rhos, target_broadcasted, reduction='none')
                    reconstruction_losses = reconstruction_losses.mean(dim=(-1, -2, -3))
                else:
                    reconstruction_losses = torch.nn.functional.mse_loss(
                        predicted_all_targets[:, i, 1:],  # skipping the first basis as it is always chosen as the first measurement
                        target.unsqueeze(1).expand(-1, predicted_all_targets.shape[2] - 1, *target.shape[1:]),
                        reduction='none'
                    )
                    if mode == 'rho':
                        reconstruction_losses = reconstruction_losses.mean(dim=(-1, -2, -3))
                    elif mode == 'concurrence':
                        reconstruction_losses = reconstruction_losses.squeeze(-1)

                if epoch < num_noisy_epochs:
                    mean_reconstruction_losses_with_noise = reconstruction_losses + torch.randn_like(reconstruction_losses) * 1e-3
                else:
                    mean_reconstruction_losses_with_noise = reconstruction_losses

                best_measurement_ids = torch.argmin(mean_reconstruction_losses_with_noise, dim=1).detach() + 1  # shifting by 1 as the first measurement is skipped
                selector_loss_i = selector_criterion(input=probabilites, target=best_measurement_ids)
                selector_loss += selector_loss_i

            selector_loss.backward()
            selector_optimizer.step()
            metrics['selector_train_loss'] += selector_loss.item()


        for _ in range(num_reconstructor_repeats):
            reconstructor_optimizer.zero_grad()
            predicted_best_targets, predicted_bases_probabilities, _ = model(measurement_with_basis, rho)
            reconstructor_loss = torch.zeros(1).to(device)
            for i in range(predicted_best_targets.shape[1]):
                if mode == 'tomo_corrections':
                    probabilites = reduce(torch.func.vmap(torch.kron), [predicted_bases_probabilities[:, i, j] for j in range(predicted_bases_probabilities.shape[2])])
                    selected_projection_vectors_ids = torch.argsort(probabilites, dim=1, descending=True)[:, :i+1]
                    selected_projection_vectors = n_qubits_projection_vectors[selected_projection_vectors_ids]
                    selected_measurements = torch.stack([m[selected_projection_vectors_ids[i]] for i, m in enumerate(measurement)])
                    inverse_corrections, r_corrections = predicted_best_targets[:, :i+1, :, 0], predicted_best_targets[:, i, :, 1]
                    inverse_corrections = torch.moveaxis(inverse_corrections, 1, -2)
                    reconstructor_loss += corrections_loss_with_dynamic_selection(reconstructor_criterion, selected_projection_vectors, gammas, target, selected_measurements, inverse_corrections, r_corrections)
                else:
                    reconstructor_loss += reconstructor_criterion(predicted_best_targets[:, i], target)

            reconstructor_loss.backward()
            reconstructor_optimizer.step()
            metrics['reconstructor_train_loss'] += reconstructor_loss.item()

        if batch_idx % log_interval == 0:
            pbar.set_postfix({'reconstructor_loss': reconstructor_loss.item(), 'selector_loss': selector_loss.item()})
    metrics['reconstructor_train_loss'] /= num_reconstructor_repeats * len(train_loader)
    metrics['selector_train_loss'] /= num_selector_repeats * len(train_loader)
    return metrics



def train_optimized_discrete_measurement_selector(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    reconstructor_optimizer: Optimizer,
    selector_optimizer: Optimizer,
    epoch: int,
    log_interval: int = 100,
    reconstructor_criterion: t.Callable = nn.MSELoss(),
    selector_criterion: t.Callable = nn.CrossEntropyLoss(),
    num_reconstructor_repeats: int = 1,
    num_selector_repeats: int = 1,
    num_noisy_epochs: int = 10
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

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train Epoch: {epoch}')
    for batch_idx, (rho, measurement, _) in pbar:
        rho, measurement = rho.to(device), measurement.to(device)
        qubits_bases_batch = qubits_bases.unsqueeze(0).expand(rho.shape[0], -1, -1, -1, -1)
        target = rho

        measurement_with_basis = [
            (measurement[:, i:i+1], qubits_bases_batch[:, i])
            for i in range(measurement.shape[1])
        ]

        for _ in range(num_selector_repeats):
            selector_optimizer.zero_grad()
            if epoch < num_noisy_epochs:
                _, predicted_bases_probabilities, predicted_basis_target_ids = model(measurement_with_basis, rho, predict_target_basis=True, add_noise_while_selecting_basis=True)
            else:
                _, predicted_bases_probabilities, predicted_basis_target_ids = model(measurement_with_basis, rho, predict_target_basis=True)
            selector_loss = torch.zeros(1).to(device)
            for i in range(1, predicted_bases_probabilities.shape[1]):
                probabilites = reduce(torch.func.vmap(torch.kron), [predicted_bases_probabilities[:, i, j] for j in range(predicted_bases_probabilities.shape[2])])
                selector_loss_i = selector_criterion(input=probabilites, target=predicted_basis_target_ids[:, i-1])
                selector_loss += selector_loss_i

            selector_loss.backward()
            selector_optimizer.step()
            metrics['selector_train_loss'] += selector_loss.item()


        for _ in range(num_reconstructor_repeats):
            reconstructor_optimizer.zero_grad()
            predicted_best_targets, predicted_bases_probabilities, _ = model(measurement_with_basis, rho)
            reconstructor_loss = torch.zeros(1).to(device)
            for i in range(predicted_best_targets.shape[1]):
                reconstructor_loss += reconstructor_criterion(predicted_best_targets[:, i], target)

            reconstructor_loss.backward()
            reconstructor_optimizer.step()
            metrics['reconstructor_train_loss'] += reconstructor_loss.item()

        if batch_idx % log_interval == 0:
            pbar.set_postfix({'reconstructor_loss': reconstructor_loss.item(), 'selector_loss': selector_loss.item()})
    metrics['reconstructor_train_loss'] /= num_reconstructor_repeats * len(train_loader)
    metrics['selector_train_loss'] /= num_selector_repeats * len(train_loader)
    return metrics
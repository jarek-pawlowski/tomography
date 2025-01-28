import typing as t
import torch


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


def regressor_balanced_accuracy(
    input: torch.Tensor,
    target: torch.Tensor,
    input_threshold: float = 0.5,
    target_threshold: float = 0.5,
    reduction: str = 'mean'
) -> torch.Tensor:
    prediction = (input > input_threshold).float()
    target = (target > target_threshold).float()
    true_positive = (prediction * target).sum()
    false_positive = (prediction * (1 - target)).sum()
    true_negative = ((1 - prediction) * (1 - target)).sum()
    false_negative = ((1 - prediction) * target).sum()
    balanced_accuracy = 0.5 * (true_positive / (true_positive + false_negative) + true_negative / (true_negative + false_positive))
    if reduction == 'mean':
        return balanced_accuracy.mean()
    return balanced_accuracy


def regressor_precision(
    input: torch.Tensor,
    target: torch.Tensor,
    input_threshold: float = 0.5,
    target_threshold: float = 0.5,
) -> torch.Tensor:
    prediction = (input > input_threshold).float()
    target = (target > target_threshold).float()
    true_positive = (prediction * target).sum()
    false_positive = (prediction * (1 - target)).sum()
    precision = true_positive / (true_positive + false_positive)
    return precision


def regressor_recall(
    input: torch.Tensor,
    target: torch.Tensor,
    input_threshold: float = 0.5,
    target_threshold: float = 0.5,
) -> torch.Tensor:
    prediction = (input > input_threshold).float()
    target = (target > target_threshold).float()
    true_positive = (prediction * target).sum()
    false_negative = ((1 - prediction) * target).sum()
    recall = true_positive / (true_positive + false_negative)
    return recall


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


def torch_bures_distance(
    rho_input: torch.Tensor,
    rho_target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    rho_input_np = torch.complex(rho_input[..., 0, :, :], rho_input[..., 1, :, :]).to(torch.cdouble)
    rho_target_np = torch.complex(rho_target[..., 0, :, :], rho_target[..., 1, :, :]).to(torch.cdouble)

    bures_distances = []
    for rho_in, rho_t in zip(rho_input_np, rho_target_np):
        try:
            fidelity = torch_fidelity(rho_in, rho_t)
        except:
            fidelity = torch.tensor(0.)
        bures_distance = 2 * (1 - torch.sqrt(fidelity))
        bures_distances.append(bures_distance.unsqueeze(0))
    bures_distances = torch.stack(bures_distances)
    if reduction == 'mean':
        return bures_distances.mean()
    return bures_distances


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


def contrastive_bases_loss(
    predicted_bases: torch.Tensor, # shape (batch_size, num_measurements, num_qubits, 2, 2)
    reduction: str = 'mean'
) -> torch.Tensor:
    predicted_bases_complex_stack = torch.stack((predicted_bases.real, predicted_bases.imag), dim=-3)
    predicted_bases_flatten = predicted_bases_complex_stack.view(predicted_bases.shape[0], predicted_bases.shape[1], -1)
    cdist = torch.cdist(predicted_bases_flatten, predicted_bases_flatten, p=2)
    loss = -cdist.mean(dim=-1).mean(dim=-1) # minus sign to convert distance to similarity so that minimizing loss maximizes differences
    if reduction == 'mean':
        return loss.mean()
    return loss


def contrastive_bases_trace_norm_loss(
    predicted_bases: torch.Tensor, # shape (batch_size, num_measurements, num_qubits, 2, 2)
    reduction: str = 'mean'
) -> torch.Tensor:
    trace_norms = []
    for measurement_idx in range(1, predicted_bases.shape[1]):
        matrix_diff = predicted_bases[:, measurement_idx:measurement_idx+1] - predicted_bases[:, :measurement_idx].detach()
        trace_norms_for_measurement = torch.linalg.matrix_norm(matrix_diff, ord=1, dim=(-2, -1)).mean(dim=-1).mean(dim=-1)
        trace_norms.append(trace_norms_for_measurement)
    trace_norms = torch.stack(trace_norms)
    loss = trace_norms
    if reduction == 'mean':
        return loss.mean()
    return loss


def complex_distance(
    input: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean',
    complex_dim: int = -3
) -> torch.Tensor:
    distances = (input - target).norm(dim=complex_dim)
    if reduction == 'mean':
        return distances.mean()
    return distances


def complex_distance_matrix_elements_avg(
    input: torch.Tensor,
    target: torch.Tensor,
    complex_dim: int = -3
) -> torch.Tensor:
    distances = complex_distance(input, target, reduction='none', complex_dim=complex_dim)
    distances = distances.view(-1, distances.shape[-2], distances.shape[-1])
    distances = distances.mean(dim=0)
    return distances

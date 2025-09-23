from math import log2
import typing as t
from functools import cache

import numpy as np
import torch
from qiskit.quantum_info import concurrence, DensityMatrix

from src.tomography_utils_numpy import Tomography, Kwiat_projectors


def tensordot(
    a: torch.Tensor,
    b: torch.Tensor,
    indices: t.Tuple[t.List[int], t.List[int]] = ([1], [0]),
    moveaxis: t.Optional[t.Tuple[int, ...]] = None,
    conj_tr: t.Tuple[bool, bool] = (False, False)
) -> torch.Tensor:

    a = torch.conj(a.transpose(-1, -2)) if conj_tr[0] else a  # warning: transposing reverses tensor indices
    b = torch.conj(b.transpose(-1, -2)) if conj_tr[1] else b  # warning: transposing reverses tensor indices
    result = torch.tensordot(a, b, indices)
    if moveaxis is not None:
        result = torch.moveaxis(result, *moveaxis)
    return result

def trace(a: torch.Tensor, batch_first: bool = False):
    # performs tensor contraction Tijk...ijk...
    a_shape = (len(a.shape) - 1) if batch_first else len(a.shape)
    dim = int(a_shape/2)
    indices = np.indices([2]*dim).reshape(dim,-1).T
    indices_to_sum = np.tile(indices, 2)
    return torch.sum(torch.stack([a[tuple(idx)] for idx in indices_to_sum]))


def measure(rho: torch.Tensor, basis_vectors: t.Tuple[torch.Tensor, ...]) -> torch.Tensor:
    '''
    Assumes:
        rho: density matrix of shape (2, 2, ..., 2), where len(shape) = num_qubits
        basis_vectors: measurement operators for each qubit; shape (num_qubits, 2, 2)
    '''
    Prho = rho.clone()
    for i, basis_vector in enumerate(basis_vectors):
        Prho = tensordot(basis_vector, Prho, indices=([1], [i]), moveaxis=(0,i))
    prob = trace(Prho).real
    return prob


def measure_batch_kron(rho: torch.Tensor, basis_vectors: torch.Tensor) -> torch.Tensor:
    '''
    Alternative batch measurement using Kronecker products.
    More memory intensive but potentially faster for small qubit counts.
    
    Args:
        rho: batch of density matrices (batch_size, 2^n, 2^n) in matrix form
        basis_vectors: batch of measurement operators (batch_size, num_qubits, 2, 2)
    
    Returns:
        prob: measurement probabilities (batch_size,)
    '''
    batch_size, num_qubits = basis_vectors.shape[:2]
    
    # Compute Kronecker product for each batch element
    # Start with first basis vector
    measurement_ops = basis_vectors[:, 0]  # (batch_size, 2, 2)
    
    # Iteratively compute Kronecker products
    for i in range(1, num_qubits):
        next_basis = basis_vectors[:, i]  # (batch_size, 2, 2)
        
        # Batch Kronecker product using einsum
        # kron(A,B)[i*n+j, k*m+l] = A[i,k] * B[j,l]
        measurement_ops = torch.einsum('bij,bkl->bikjl', measurement_ops, next_basis)
        measurement_ops = measurement_ops.reshape(batch_size, -1, measurement_ops.shape[-1] * measurement_ops.shape[-2])
    
    # Compute trace(measurement_op @ rho) for each batch
    # Using tr(AB) = sum(A * B^T) element-wise
    prob = torch.sum(measurement_ops * rho.transpose(-2, -1), dim=(-2, -1)).real
    
    return prob


def my_measure(rho: torch.Tensor, basis_vectors: t.Tuple[torch.Tensor, ...], basis_batched: bool = False) -> torch.Tensor:
    # measure all qubits using list of operators
    # basis_vectors = operators to use when measuring subsequent qubits
    Prho = rho.clone()
    if basis_batched:
        basis_vectors = torch.moveaxis(basis_vectors, 0, 1)
        batch_correction = 1
    else:
        batch_correction = 0
    # batch_idx = 1 if batch_first else 0
    for i, basis_vector in enumerate(basis_vectors):
        Prho = tensordot(basis_vector, Prho, indices=([1 + batch_correction], [i]), moveaxis=(batch_correction, i + batch_correction))
    prob = trace(Prho).real
    return prob


def calculate_concurrence_from_measurements(measurements_data: t.Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    num_qubits = 2 # tested for 2 qubits
    dim = 2**num_qubits
    tomography = Tomography(num_qubits, Kwiat_projectors)
    tomography.calulate_B_inv()

    predictions = []
    for measurements in measurements_data:
        if isinstance(measurements, np.ndarray):
            rho_rec = tomography.reconstruct(measurements, enforce_positiv_sem=True)
        elif isinstance(measurements, torch.Tensor):
            rho_rec = tomography.reconstruct(measurements.cpu().numpy(), enforce_positiv_sem=True)
        else:
            raise ValueError('measurements must be numpy array or torch tensor')
        rho_rec = rho_rec.reshape((dim, dim))
        rho_rec = rho_rec / np.trace(rho_rec)
        density_matrix = DensityMatrix(rho_rec)
        try:
            conc = concurrence(density_matrix)
        except:
            conc = -1
        predictions.append(conc)
    if isinstance(measurements_data, np.ndarray):
        predictions = np.array(predictions)
        return predictions
    predictions = torch.tensor(predictions).unsqueeze(-1)
    return predictions.to(measurements_data.device)


def reconstruct(measurements: torch.Tensor, projection_vectors: torch.Tensor, gammas: torch.Tensor, enforce_valid_density_matrix: bool = True, inverse: str = 'exact', zero_measurements: t.Optional[t.List[int]] = None):
    """
    Torch tomography reconstruction method
    
    Args:
        measurements: tensor of shape (num_measurements)
        projection_vectors: tensor of shape (num_measurements, dim, 1)
            where dim = 2**num_qubits
        gammas: matrices used to reconstruct the density matrix, 
            e.g. pauli matrices, shape (num_gammas, dim, dim), 
            where dim = 2**num_qubits
        enforce_valid_density_matrix: if True, the reconstructed density matrix will be made hermitian,
            normalized and positive semidefinite
    """
    B = calculate_B(projection_vectors, gammas).to(measurements.device)
    if inverse == 'exact':
        B_inv = torch.linalg.inv(B)
        if zero_measurements is not None:
            B_inv[:, zero_measurements] = 0
    elif inverse == 'pinv':
        B_inv = torch.linalg.pinv(B)
    else:
        raise ValueError(f'Unknown inverse method: {inverse}')
    r = torch.matmul(B_inv, measurements.to(torch.complex64))
    rho = tensordot(gammas, r, indices=([0], [0]))
    if enforce_valid_density_matrix:
        # make rho hermitian
        rho = (rho + torch.conj(rho.T)) / 2
        # normalize to trace 1
        if rho.trace() == 0:
            raise ValueError('Trace of density matrix is zero')
        rho = rho / rho.trace()
        # make rho positive semidefinite
        eigs = torch.amin(torch.linalg.eigvalsh(rho))
        if eigs < 0.: rho -= torch.eye(rho.shape[-1])*eigs   
    return rho


def calculate_B(projection_vectors: torch.Tensor, gammas: torch.Tensor):
    # Compute conjugate of projection_vectors
    projection_vectors_conj = torch.conj(projection_vectors)
    
    # Use einsum for batch tensor operations
    intermediate_result = torch.einsum('ijk,nkm->injm', gammas, projection_vectors)
    B = torch.einsum('njm,injm->ni', projection_vectors_conj, intermediate_result)
    return B

@cache
def calculate_B_inv(projection_vectors: torch.Tensor, gammas: torch.Tensor, method: str = 'exact'):
    B = calculate_B(projection_vectors, gammas)
    if method == 'exact':
        B_inv = torch.linalg.inv(B)
    elif method == 'pinv':
        B_inv = torch.linalg.pinv(B)
    else:
        raise ValueError(f'Unknown inverse method: {method}')
    return B_inv

def reconstruct_with_nn_corrections(
    measurements: torch.Tensor,
    projection_vectors: torch.Tensor,
    gammas: torch.Tensor,
    inverse_correction: torch.Tensor,
    r_correction: torch.Tensor,
    m2_corrections: t.Optional[torch.Tensor] = None
):
    ''' 
    assumes:
        inverse_correction is a tensor of shape (num_measurements, num_gammas)
        r_correction is a tensor of shape (num_gammas)
        m2_corrections is a tensor of shape (num_measurements, num_measurements)
    '''
    with torch.no_grad():
        B = calculate_B(projection_vectors, gammas)
        B_inv = torch.linalg.pinv(B)

    B_inv = B_inv.detach().to(inverse_correction.device) + inverse_correction.T
    r = torch.matmul(B_inv, measurements.to(torch.complex64)) + r_correction

    if m2_corrections is not None:
        m2 = torch.prod(torch.combinations(measurements, 2, with_replacement=True), dim=-1).to(torch.complex64)
        r = r + m2 @ m2_corrections

    rho = tensordot(gammas, r, indices=([0], [0]))
    return rho


def reconstruct_with_nn_corrections_and_B_inv(
    measurements: torch.Tensor,
    B_inv: torch.Tensor,
    gammas: torch.Tensor,
    inverse_correction: torch.Tensor,
    r_correction: torch.Tensor
):
    ''' 
    assumes:
        inverse_correction is a tensor of shape (num_measurements, num_gammas)
        r_correction is a tensor of shape (num_gammas)
    '''
    B_inv_corr = B_inv + inverse_correction.T
    r = torch.matmul(B_inv_corr, measurements.to(torch.complex64)) + r_correction
    rho = tensordot(gammas, r, indices=([0], [0]))
    return rho

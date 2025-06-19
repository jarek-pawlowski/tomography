from math import log2
import typing as t

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
    
    a = torch.conj(a.T) if conj_tr[0] else a  # warning: transposing reverses tensor indices
    b = torch.conj(b.T) if conj_tr[1] else b  # warning: transposing reverses tensor indices
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

def measure(rho: torch.Tensor, basis_vectors: t.Tuple[torch.Tensor, ...]) -> torch.Tensor:
    # measure all qubits using list of operators
    # basis_vectors = operators to use when measuring subsequent qubits
    
    Prho = rho.clone()
    for i, basis_vector in enumerate(basis_vectors):
        Prho = tensordot(basis_vector, Prho, indices=([1], [i]), moveaxis=(0,i))
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

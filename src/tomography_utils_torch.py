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


def calculate_concurrence_from_measurements(measurements_data: torch.Tensor) -> torch.Tensor:
    num_qubits = 2 # tested for 2 qubits
    dim = 2**num_qubits
    tomography = Tomography(num_qubits, Kwiat_projectors)
    tomography.calulate_B_inv()

    predictions = []
    for measurements in measurements_data:
        rho_rec = tomography.reconstruct(measurements.cpu().numpy(), enforce_positiv_sem=True)
        rho_rec = rho_rec.reshape((dim, dim))
        rho_rec = rho_rec / np.trace(rho_rec)
        density_matrix = DensityMatrix(rho_rec)
        try:
            conc = concurrence(density_matrix)
        except:
            conc = -1
        predictions.append(conc)
    predictions = torch.tensor(predictions).unsqueeze(-1)
    return predictions.to(measurements_data.device)


def reconstruct(measurements: torch.Tensor, projection_vectors: torch.Tensor, gammas: torch.Tensor, enforce_valid_density_matrix: bool = True, inverse: str = 'exact', zero_measurements: t.Optional[t.List[int]] = None):
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
    # num_projection_vectors = projection_vectors.shape[0]
    # num_gammas = gammas.shape[0]
    # B_old = torch.zeros((num_projection_vectors, num_gammas), dtype=torch.complex64) #, device=gammas.device)
    # for nu in range(num_projection_vectors):
    #     for mu in range(num_gammas):
    #         B_old[nu,mu] = tensordot(projection_vectors[nu], tensordot(gammas[mu], projection_vectors[nu]), conj_tr=(True,False)).item()
    # # return B

    # num_projection_vectors = projection_vectors.shape[0]
    # num_gammas = gammas.shape[0]
    # B_old2 = torch.zeros((num_projection_vectors, num_gammas), dtype=torch.complex64) #, device=gammas.device)
    # for nu in range(num_projection_vectors):
    #     for mu in range(num_gammas):
    #         B_old2[nu,mu] = torch.matmul(projection_vectors[nu].conj().T, torch.matmul(gammas[mu], projection_vectors[nu])).item()


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
    r_correction: torch.Tensor
):
    ''' 
    assumes:
        inverse_correction is a tensor of shape (num_measurements, num_gammas)
        r_correction is a tensor of shape (num_gammas)
    '''
    with torch.no_grad():
        B = calculate_B(projection_vectors, gammas)
        B_inv = torch.linalg.pinv(B)

    B_inv = B_inv.detach().to(inverse_correction.device) + inverse_correction.T
    r = torch.matmul(B_inv, measurements.to(torch.complex64)) + r_correction
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

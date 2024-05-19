import numpy as np
import functools as ft

import src.utils_measure as utils


number_qubits = 3
space_size = int(pow(2, number_qubits))
measurement = utils.Measurement(utils.Pauli, number_qubits, basis_c=utils.Pauli_c)
 
#psi_in = np.ones((2,2,2), dtype=complex)/2./np.sqrt(2.) #reshape do tensora 2x2x2
# test dla Bella
psi_in = np.zeros((2,2,2), dtype=complex)
psi_in[0,0,0] = 1./np.sqrt(2.)
psi_in[1,1,1] = 1./np.sqrt(2.)


T = 10000
snapshots = []

for t in range(T):
    basis = []
    measurement_shadow = []
    p = psi_in.copy() 
    for i in range(number_qubits):
        basis_index = np.random.randint(len(measurement.basis))
        m, p = measurement.measure_single_pure(p, i, basis_index = basis_index, return_state=True)
        basis.append(basis_index)
        measurement_shadow.append(m)
    snapshots.append([basis, measurement_shadow])
    
# now try to reconstruct the state
#psi_in = do rho0 from psi_in
rho0 = utils.tensordot(psi_in, psi_in, indices=0, conj_tr=(False,True))
rho1 = np.zeros_like(rho0.reshape(space_size,space_size))

for it in range(T):
    basis, measurement_shadow  = snapshots[it]
    single_rho_collection = []
    for iq in range(number_qubits):
        single_rho_collection.append((np.eye(2)+measurement_shadow[iq]*utils.Pauli_vector[basis[iq]]*3)/2.)  # (Eq. A5) in https://arxiv.org/pdf/2106.12627.pdf
    rho1 += ft.reduce(np.kron, single_rho_collection)  # Kronecker product of (multiple) single-qubit rhos 
rho1 /= T
    
np.set_printoptions(linewidth=200)
print(np.around(rho0.reshape(space_size,space_size), 3))
print(np.around(rho1, 3))

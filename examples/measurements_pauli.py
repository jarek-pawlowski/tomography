import numpy as np
import functools as ft
import matplotlib.pyplot as plt

import src.utils_measure as utils

def measure_shadow(T):
    
    snapshots = []

    for t in range(T):
        basis = []
        measurement_shadow = []
        p = psi_in.copy() 
        for i in range(number_qubits):
            basis_index = np.random.randint(len(measurement.basis))
            m, p = measurement.measure_single_pure(p, i, basis_index = basis_index, return_state=True)
            random = np.random.random()
            basis.append(basis_index)
            if m > random: 
                measurement_shadow.append(1.)
            else:
                measurement_shadow.append(-1.)
        snapshots.append([basis, measurement_shadow])
        
    return snapshots

def reconstruct_from_shadow(T, rho0, snapshots, space_size):
    
    rho1 = np.zeros_like(rho0.reshape(space_size,space_size))

    for it in range(T):
        basis, measurement_shadow  = snapshots[it]
        single_rho_collection = []
        for iq in range(number_qubits):
            single_rho_collection.append((np.eye(2)+measurement_shadow[iq]*utils.Pauli_vector[basis[iq]]*3)/2.)  # (Eq. A5) in https://arxiv.org/pdf/2106.12627.pdf
        rho1 += ft.reduce(np.kron, single_rho_collection)  # Kronecker product of (multiple) single-qubit rhos 
    rho1 /= T
    
    return rho1

'''
number_qubits = 3
space_size = int(pow(2, number_qubits))
measurement = utils.Measurement(utils.Pauli, number_qubits)
              
psi_in = np.ones((2,2,2), dtype=complex)/2./np.sqrt(2.) #reshape to 2x2 tensor (each 2x2 matrix describes a single qubit)

rho0 = utils.tensordot(psi_in, psi_in, indices=0, conj_tr=(False,True))

T = 10000
snapshots = measure_shadow(T)
rho1 = reconstruct_from_shadow(T, rho0, snapshots, space_size)
    
# now try to reconstruct the state
#psi_in = do rho0 from psi_in

np.set_printoptions(linewidth=200)
print(np.around(rho0.reshape(space_size,space_size), 3))
print("\n")
print(np.around(rho1, 3))
'''

number_qubits = 2
space_size = int(pow(2, number_qubits))
measurement = utils.Measurement(utils.Pauli, number_qubits)
psi_in = np.ones((2,2), dtype=complex)/2./np.sqrt(2.) #reshape to 2x2 tensor (each 2x2 matrix describes a single qubit)
rho0 = utils.tensordot(psi_in, psi_in, indices=0, conj_tr=(False,True))

set_of_T = [100, 1000, 10000, 100000]
norms = []
for t in set_of_T: 
    print(f"Start {t}")
    snapshots = measure_shadow(t)
    rho1 = reconstruct_from_shadow(t, rho0, snapshots, space_size)
    norms.append(np.linalg.norm(rho0.reshape(space_size,space_size)-rho1.reshape(space_size,space_size)))
    
plt.plot(set_of_T, norms, '-o')
plt.xscale('log')
plt.yscale('log')
plt.show()
    
    

#test dla bella potem 
# test for Bell state:
rho_in = np.zeros((4,4))
rho_in[0,0] = 0.5
rho_in[0,3] = 0.5
rho_in[3,0] = 0.5
rho_in[3,3] = 0.5
m_all = np.array([[measurement.measure(rho_in.reshape((2,2,2,2)), [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()

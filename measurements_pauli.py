import utils_measure as utils

import numpy as np
import functools as ft

number_qubits = 2 
measurement = utils.Measurement(utils.Pauli, number_qubits)
        
print("the same for pure state:")        
psi_in = np.ones((2,2), dtype =complex)/2. #reshape do tensora 2x2x2
print(psi_in)

T = 10000
snapshot = []
#shadow = np.zeros(shape = (T,number_qubits))

p = psi_in 
for t in range(T):
    basis = []
    measurement_shadow = []
    for i in range(number_qubits): 
        basis_index = np.random.randint(3)
        m, p = measurement.measure_single_pure(p, i, basis_index = basis_index, return_state=True)
        random = np.random.random()
        basis.append(basis_index)
        if m > random: 
            #print("1", basis_index)
            measurement_shadow.append(1)
            #shadow[t,i] = [1, basis_index]
        else:
            #print("-1", basis_index)
            #shadow[t,i] = [-1, basis_index]
            measurement_shadow.append(-1)
        snapshot.append([basis, measurement_shadow])

# Pauli vector
sx = np.array([[ 0. , 1. ],[ 1. , 0. ]])
sy = np.array([[ 0. ,-1.j],[ 1.j, 0. ]])
sz = np.array([[ 1. , 0. ],[ 0. ,-1. ]])
pauli_vector = [sx, sy, sz]
    
# now try to reconstruct the state
#psi_in = do rho0 from psi_in
rho0 = utils.tensordot(psi_in, psi_in, indices=0,conj_tr=(False,True))
#rho1 = np.zeros_like(rho0.data)
#print(rho0.reshape(4,4))
rho0_a = np.matmul(psi_in.reshape(1,4).T,np.conj(psi_in.reshape(1,4)))
#print("this is rho0_a: " ,rho0_a)
rho1 = np.zeros_like(rho0_a.reshape(4,4))

for it in range(T):
    basis, measurement_shadow  = snapshot[it] #0 -1 
    #print(snapshot[it])
    single_rho_collection = []
    for iq in range(number_qubits): #check if right size
        #print(pauli_vector[basis[iq]])
        single_rho_collection.append((np.eye(2)+measurement_shadow[iq]*pauli_vector[basis[iq]]*3)/2.)  # (Eq. A5) in https://arxiv.org/pdf/2106.12627.pdf
    rho1 += ft.reduce(np.kron, single_rho_collection)  # Kronecker product of (multiple) single-qubit rhos 
rho1 /= T
    
np.set_printoptions(linewidth=200)
print(np.around(rho0_a, 3))
print(np.around(rho1, 3))

#test dla bella potem 
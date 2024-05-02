import numpy as np
import functools as ft
import torch
from src.datasets import MeasurementVectorDataset
from src.torch_measure import Measurement, tensordot
from src.utils_measure import Pauli, Pauli_c, Pauli_vector
import matplotlib.pyplot as plt

# Define the path to the training_states directory
training_states_dir = './training_states/'
num_qubits = 2
space_size = int(pow(2, num_qubits))

train_dataset = MeasurementVectorDataset(num_qubits, root_path=training_states_dir, return_density_matrix=True)

idx = 0
vector = train_dataset[idx]
print(vector[0])
print(vector[1])
print(vector[0][0])
print(vector[0][1])

measurement = Measurement(num_qubits)
basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli.basis]
basis_matrices_c = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli_c.basis] #complementary basis 
Pauli_basis_vector = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli_vector] #3 normal Pauli matrices

snapshot = []
psi_in = torch.complex(vector[0][0], vector[0][1]).view(*[2]*num_qubits) #because its a vector
rho0 = tensordot(psi_in, psi_in, indices=0, conj_tr=(True,False)) #torch tensordot

print(f"This is psi_in {psi_in}")
print(rho0)
T = 1000
#snapshots = measure_shadow_torch(T, num_qubits, psi_in, measurement)
#rho1 = reconstruct_from_shadow_torch(T, rho0, snapshots, space_size, num_qubits)

snapshot = []

basis_size = len(Pauli.basis)
basis_reconstruction = torch.stack(basis_matrices, dim=0)


for i in range(T):
    measurement_shadow = [] 
    basis_vectors = []
    for i in range(num_qubits):
        basis_index = np.random.randint(basis_size)
        m, p = measurement.measure_single_pure(psi_in, i, basis_matrices[basis_index], basis_matrices_c[basis_index], return_state = True)
        basis_vectors.append(basis_index)
        measurement_shadow.append(m)
    
    snapshot.append([basis_vectors, measurement_shadow])

#print(snapshot)
rho1 = torch.zeros_like(rho0.view(space_size, space_size))

for it in range(T):
    basis, measurement_shadow  = snapshot[it]
    single_rho_collection = []
    for iq in range(num_qubits):
        single_rho_collection.append((torch.eye(2) + measurement_shadow[iq] * Pauli_basis_vector[basis[iq]] * 3) / 2.)  # (Eq. A5) in https://arxiv.org/pdf/2106.12627.pdf
    rho1 += ft.reduce(torch.kron, single_rho_collection)  # Kronecker product of (multiple) single-qubit rhos 

rho1 /= T


    
np.set_printoptions(linewidth=200)
print("rho initial: ")
print(rho0.reshape(space_size,space_size))
print("\n")
print("rho reconstructed: ")
print(rho1)
#print(np.around(rho1, 3))
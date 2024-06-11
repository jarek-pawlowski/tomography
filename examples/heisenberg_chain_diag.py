# version 3.0 for spin 1D chain diagonalization (max number of sites 8-10) using numpy and:
# -> calculating a density matrix
# -> calculating a reduced density matrix of chain divided into two equal subsytems
# -> calculating entropy of this subsystems
# -> making a matrix block diagonal by simple matrix operations
# -> calculation of rho reduced density matrices with fixed s_z quantum number
####

import numpy as np
from functools import reduce
from itertools import chain, product
import os 
import math
import src.utils_measure as utils
import src.utils_entropy as entropy_calc
#my .py files

######## Heisenberg Graph #########
'''                                                                    
Program for calculating and diagonalization of Heisenberg Hamiltonian for graph with defined adjacency matrix                               # 
# class Graph - define graph, class Heisenberg - calculation of H
# here you can creating a graph using add_edge methods or declare prepared matrix (adjMatrix).                                              #
# N is a size of the system # 
'''
#sites = [4,6,8,10]
#sites = [4,6,7,8,9]
#for i in sites:
        #in this code it's enough to define one "hopping", becasue the second one is already implemented in the code
        #make above note more precise! 
        
        #4 sites open
        #adjMatrix = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
        
size_of_the_chain = 12
adjMatrix = np.eye(size_of_the_chain, k=1, dtype=int)[::]
adjMatrix[-1][0] = 0
#above matrix for size = 4 is : #adjMatrix = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
print("This is adjacency Matrix : \n", adjMatrix)

S = 1/2 #spin number 
N = len(adjMatrix) #size of the system
print("Calculating N = " + str(N)  + " system for S = " + str(S))

H = entropy_calc.Heisenberg(N, S)
print("Start the diagonalization")

basis_H, basis_H_s_z, spins = H.calculate_basis()

#J a random number form 0.01 to 1.0 
J = 1.0
#J = np.random.uniform(-1.0, 1.0)
print(f"J equals = {J}")
H.create_Hamiltonian(J, adjMatrix)

energies, vectors = H.eig_diagonalize_Heisenberg()
#print(energies)
#find a ground state and save its coresponding eigenvector as a tensor
#ground_state_index = np.argmin(energies)
#ground_state_energy = energies[ground_state_index]
#ground_state_vector = vectors[:, ground_state_index#]

#choose ten eigenvalues with the lowest energy and their corresponding eigenvectors
energies = energies[:10]
vectors = vectors[:,:10]
#print(energies)

#tensor = ground_state_vector.reshape(tuple(2 for _ in range(size_of_the_chain)))
#filename = f'./training_states/{size_of_the_chain}_tensor_ground.npy'
#np.save(filename, tensor)

folder_name = f'training_states/training_states_{size_of_the_chain}'

with open (f'{folder_name}/dictionary.txt', 'w') as file:
    
    for j in range(len(energies)):
        
        #print(f"Eigenvectors for this spin: {vectors[:,j]}")
        #print(f"This is energy {energies[j]}")   
                
        # Reshape the vector into a 2x2 tensor
        tensor = vectors[:,j].reshape(tuple(2 for _ in range(size_of_the_chain)))
        #Jarek's tensordot used in the shadow calculations
        # density_matrix = utils.tensordot(vectors[:,j], vectors[:,j], indices=0, conj_tr=(True,True))
        
        # Now, save this tensor to a .npy file
        #filename = f'./{folder_name}/train/{size_of_the_chain}_tensor_state_{j}.npy'
        filename = f'./{folder_name}/{size_of_the_chain}_tensor_state_{j}.npy'
        file.write(f"{J} {energies[j]}\n")
        np.save(filename, tensor)

#print(tensor)
# To verify, let's load the file and print the tensor
#loaded_tensor = np.load(f'./training_states/train/{size_of_the_chain}_tensor_state_{j}.npy')
#print(loaded_tensor)
#print(loaded_tensor.dtype)




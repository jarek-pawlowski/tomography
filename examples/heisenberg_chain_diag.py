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
#my .py files

class Heisenberg(object):

    #Creating Heisenberg Hamiltonian 

    #Initialization of the system and spin opearators
    def __init__(self,N, S, directory = None) -> None:
        self.size_of_system = N
        self.chain_I = []
        self.S_site_whole = []
        self.energies = []
        self.vectors = []
        self.possible_basis = []
        self.H = 0
        self.S = S
        self.basis = []
        self.list_of_spins = []
        self.list_spins = []
        self.basis_s_z = []
        
        if S == 1:
        #matrices for S = 1
            self.S_plus = np.sqrt(2) * np.array([[0,1,0],
                                            [0,0,1],
                                            [0,0,0]])

            self.S_minus = np.sqrt(2) * np.array([[0,0,0],
                                            [1,0,0],
                                            [0,1,0]])
            self. S_z = np.array([[1,0,0],
                            [0,0,0],
                            [0,0,-1]])
            
            self.I = np.array([[1,0,0],
                [0,1,0],
                [0,0,1]])
        
        
        elif S == 1/2:
            #matrices for S = 1/2
            self.S_plus = np.array([[0,1],
                                    [0,0]])

            self.S_minus = np.array([[0,0],
                                    [1,0]])
        
            self.S_z = 1/2 * np.array([[1,0],
                                    [0,-1]])
            self.I = np.array([[1,0],
                                [0,1]])
    
    def S_site(self, index, S):
        #Using tensor product to calculate S_i matrix
        N = self.size_of_system - 1
        self.chain_I = chain([np.identity(len(S)**(index))], [S], [np.identity(len(S)**(N - index))])
        return reduce(np.kron, self.chain_I)
    
    def S_z_operator(self):
        #calculating S_z operator as sum S_z_1 + S_z_2 + ... + S_z_N
        S_z_operator = 0
        #print(self.S_z)
        for i in range(self.size_of_system+1):
            S_z_operator  += self.S_site(i, self.S_z)
        
        #print(S_z_operator)
        return S_z_operator
        
    def calc_Sz(self, eigenvector):
        # Calculate the conjugate transpose of the eigenvector
        eigen_dagger = np.conj(eigenvector.T)
        # Calculate the expectation value of S_z
        Sz_total = np.dot(eigen_dagger, np.dot(self.S_z_operator(), eigenvector))
        return Sz_total
    
    def calculate_basis(self):
        N = self.size_of_system
        #for bais s=1/2 -> (up - True, down - False)
        #for bais s=1 -> (-1,0,1)
        
        if self.S == 1/2: 
            self.list_spins = [1/2,-1/2]
        elif self.S == 1:
            self.list_spins = [-1,0,1]
            
        for i in range(N):
            self.possible_basis.append(self.list_spins)
            
        #whole basis
        #basis_s_z = []
        self.basis = list(product(*self.possible_basis))
        self.basis_s_z = self.basis[:]
        
        #self.basis = list(map(lambda x: list(x), self.basis))
        #print(self.basis)
        
        for i in range(len(self.basis_s_z)):
            self.basis_s_z[i] = sum(self.basis_s_z[i])
            
        #print(self.basis)
        #all possible spin combinations
        self.list_of_spins = sorted(list(set(self.basis_s_z)),reverse=True)
        #print(self.list_of_spins)
        
        return self.basis, self.basis_s_z, self.list_of_spins 
    
    
    def subsystems_fixed_s_z(self, spin_basis,size_of_sub_A,size_of_sub_B):
        #function for calculating bases of subsystems A and B
        # here is posible problem that: 
        # if size of subA = 3
        # and size of subB = 3, but whole system is 6 sties then: 
        # 6/3 = 2
        # spin_basis[:(6/3)] = last 2 elements
        # spin_basis[(6/3):] = first 4 elements
        # if you want to have a proper division divide bases equally!
        
        #DIVISION OF A AND B HERE IS VERY CRUCIAL FOR THE RHO CALCULATION 
        #CHECK 135 - 139 lines and 160 - 164
        
        subsystem_A = list(set(map(lambda x: x[:size_of_sub_A], spin_basis)))
        subsystem_B = list(set(map(lambda x: x[size_of_sub_B:], spin_basis)))
   
        subsystem_A_beta = list(map(lambda x: x[:size_of_sub_A], spin_basis))
        subsystem_B_beta = list(map(lambda x: x[size_of_sub_B:], spin_basis))
        
    
        #subsystem_A = list(set(map(lambda x: x[:int(len(x)/size_of_sub_A)], spin_basis)))
        #subsystem_B = list(set(map(lambda x: x[int(len(x)/size_of_sub_B):], spin_basis)))
   
        #subsystem_A_beta = list(map(lambda x: x[:int(len(x)/size_of_sub_A)], spin_basis))
        #subsystem_B_beta = list(map(lambda x: x[int(len(x)/size_of_sub_B):], spin_basis))
        
        print(subsystem_A_beta)
        print(subsystem_B_beta)
        print(f"Basis for subsystem A: {subsystem_A}")
        print(f"Basis for subsystem B: {subsystem_B}")
    
        new_basis = []
        
        for k in spin_basis:
            #k_A = k[:int(len(k)/size_of_sub_A)]
            #print(f"This k_A {k_A}")
            #k_B = k[int(len(k)/size_of_sub_B):]
            
            k_A = k[:size_of_sub_A]
            #print(f"This k_A {k_A}")
            k_B = k[size_of_sub_B:]
            #print(f"This k_B {k_B}")
        
            i = subsystem_A.index(k_A)
            j = subsystem_B.index(k_B)
            #print(f"This is i {i}")
            #print(f"This is j {j}")
            if (i,j) not in new_basis:
             new_basis.append((i,j))
    
        print("This is new basis: ", new_basis)

        return subsystem_A, subsystem_B, new_basis
    
    
    def create_Hamiltonian(self, J , adjMatrix):
        #definition of S matrices and diagonalization

        #using adjacency matrix to define neighbouring sites
        for i in range(len(adjMatrix)):
            for j in range(len(adjMatrix)):
                if adjMatrix[j][i] == 1:
                    self.H += 1/2 * J * (np.dot(self.S_site(j, self.S_plus),self.S_site(i, self.S_minus)) \
                    + np.dot(self.S_site(j, self.S_minus),self.S_site(i, self.S_plus))) \
                    + np.dot(self.S_site(j, self.S_z), self.S_site(i, self.S_z))
        
        #for i in range(len(self.H)):
                #self.H[i][i] += np.random.random()*10e-8
        #print(self.H)
        print("Len of Hamiltonian: ", len(self.H))
        
        
    def eig_diagonalize(self,A):
        #fucntion for diagonalization with sorting eigenvalues and rewriting eigenvectors as a list
        eigenValues, eigenVectors = np.linalg.eig(A)
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors
    
    def eig_diagonalize_Heisenberg(self):
        #fucntion for diagonalization with sorting eigenvalues and rewriting eigenvectors as a list
        eigenValues, eigenVectors = np.linalg.eig(self.H)
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors
    
    def block_Hamiltonian(self,iterator):
        
        block_H_spin_list = []
        #print("Look for spin: ", self.list_of_spins[iterator])
        spin_basis = []
        
        for i in range(len(self.basis_s_z)):
            if self.list_of_spins[iterator] == self.basis_s_z[i]:
                block_H_spin_list.append(self.H[i])
                spin_basis.append(self.basis[i])
            
        block_H_spin = np.vstack(block_H_spin_list)
        block_H_spin = block_H_spin[:,~np.all(block_H_spin == 0, axis = 0)]
        
        #print(f"this is  a block for {iterator} \n ", block_H_spin)
        #print(f"Is symmetric?", np.allclose(block_H_spin, block_H_spin.T, rtol=10e-5, atol=10e-8))
        
        #print(f"This is spin_basis {spin_basis}")
        
        energies, vectors = self.eig_diagonalize(block_H_spin)
        
        return energies, vectors, spin_basis
    
    def calculate_rho_system(self,psi0):
        #old functions for calulating rho for H not divdied into fixed S_z blocks
        
        #(2S+1)**N_sys
        size_of_subsystem = len(self.I)**(int(self.size_of_system/2)) 
        psi0 = psi0.reshape([size_of_subsystem, -1], order="C")
        #print("2 This is psi0: \n", psi0)
        rho = np.dot(psi0, psi0.conj().transpose())
        
        return rho
    
    def calculate_rho_env(self,psi0):
        #old functions for calulating rho for H not divdied into fixed S_z blocks
        
        size_of_subsystem = len(self.I)**(int(self.size_of_system/2))  
        psi0 = psi0.reshape([size_of_subsystem, -1], order="C")
        #print("2 This is psi0: \n", psi0)
        rho = np.dot(psi0.conj().transpose(), psi0)
    
        return rho
    
    def calculate_entropy(self,rho_reduced,n):
        
        #Here depending if s = 1/2 or s = 1 you need to change the base of log 
        
        #n - number of spins in the subsystem
        eigen_rho, vectors = self.eig_diagonalize(rho_reduced) 
        
        #entropy = -sum(eigen_rho*np.log(eigen_rho, where=0<eigen_rho, out=0.0*eigen_rho))
        #eigen_rho_nonzero = eigen_rho[(eigen_rho > 10e-8) & (eigen_rho < 1.0)]
        #entropy = -np.sum(eigen_rho_nonzero * np.log2(eigen_rho_nonzero))
        
        entropy = 0
        for i in range(len(eigen_rho)):
            #print(eigen_rho[i])
            if eigen_rho[i] <= 10e-8:
                entropy += 0.0
                
            elif eigen_rho[i] == 1.0:
                entropy += 0.0
                
            else:
                entropy += -(eigen_rho[i]*np.log2(eigen_rho[i]))
               #entropy += -(eigen_rho[i]*math.log(eigen_rho[i],3))
        
        #return entropy, eigen_rho
        
        return entropy/(n*np.log2(n)), eigen_rho
        
        #return entropy/(n*math.log(n,3)), eigen_rho
    
    def calculate_S_z(self,vectors): 
        #S_z value calculated as inner product of S_z operator and eigenvectors of H
        S_z_total = []
        for i in range(len(vectors)):
            S_z_total.append(self.calc_Sz(vectors[:,i]))
            
        return S_z_total
    

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
        
size_of_the_chain = 6
adjMatrix = np.eye(size_of_the_chain, k=1, dtype=int)[::]
adjMatrix[-1][0] = 0
#above matrix for size = 4 is : #adjMatrix = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
print("This is adjacency Matrix : \n", adjMatrix)

S = 1/2 #spin number 
N = len(adjMatrix) #size of the system
print("Calculating N = " + str(N)  + " system for S = " + str(S))

H = Heisenberg(N, S)
print("Start the diagonalization")

basis_H, basis_H_s_z, spins = H.calculate_basis()

#J a random number form 0.01 to 1.0 
J = 1.0
#J = np.random.uniform(-1.0, 1.0)
print(f"J equals = {J}")
H.create_Hamiltonian(J, adjMatrix)

energies, vectors = H.eig_diagonalize_Heisenberg()
print(energies)
#find a ground state and save its coresponding eigenvector as a tensor
#ground_state_index = np.argmin(energies)
#ground_state_energy = energies[ground_state_index]
#ground_state_vector = vectors[:, ground_state_index#]

#choose ten eigenvalues with the lowest energy and their corresponding eigenvectors
energies = energies[:10]
vectors = vectors[:,:10]
print(energies)

#tensor = ground_state_vector.reshape(tuple(2 for _ in range(size_of_the_chain)))
#filename = f'./training_states/{size_of_the_chain}_tensor_ground.npy'
#np.save(filename, tensor)

folder_name = f'./training_states_{size_of_the_chain}'

with open (f'{folder_name}/dictionary.txt', 'w') as file:
    
    for j in range(len(energies)):
        
        #print(f"Eigenvectors for this spin: {vectors[:,j]}")
        #print(f"This is energy {energies[j]}")   
                
        # Reshape the vector into a 2x2 tensor
        tensor = vectors[:,j].reshape(tuple(2 for _ in range(size_of_the_chain)))
        #Jarek's tensordot used in the shadow calculations
        # density_matrix = utils.tensordot(vectors[:,j], vectors[:,j], indices=0, conj_tr=(True,True))
        
        # Now, save this tensor to a .npy file
        filename = f'./{folder_name}/train/{size_of_the_chain}_tensor_state_{j}.npy'
        file.write(f"{J} {energies[j]}\n")
        np.save(filename, tensor)

print(tensor)
# To verify, let's load the file and print the tensor
loaded_tensor = np.load(f'./training_states/train/{size_of_the_chain}_tensor_state_{j}.npy')
print(loaded_tensor)
print(loaded_tensor.dtype)




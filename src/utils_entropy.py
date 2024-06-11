import numpy as np
from functools import reduce
from itertools import chain, product
import os 
import math


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
    
    def sz_subspace(self,iterator):
        
        spin_basis = []
        
        for i in range(len(self.basis_s_z)):
            if self.list_of_spins[iterator] == self.basis_s_z[i]:
                spin_basis.append(self.basis[i])
        
        
        return spin_basis
    
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
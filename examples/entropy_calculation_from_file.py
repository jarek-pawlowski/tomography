# version 3.0 for spin 1D chain diagonalization (max number of sites 8-10) using numpy and:
# -> calculating a density matrix
# -> calculating a reduced density matrix of chain divided into two equal subsytems
# -> calculating entropy of this subsystems
# -> making a matrix block diagonal by simple matrix operations
# -> calculation of rho reduced density matrices with fixed s_z quantum number
####


import src.utils_measure as utils
import src.utils_entropy as entropy_calc
import numpy as np
import pandas as pd
#my .py files


size_of_the_chain = 21

# Open the file and read the lines
with open(f'datasets/Eigenvalues_results_{size_of_the_chain}_feast.dat', 'r') as f:
    lines = f.readlines()
# Skip the first line (comment) and extract the first column (eigenvalues)
eigenvalues = [float(line.split(',')[0]) for line in lines[1:]]
# Convert the list to a numpy array
eigenvalues = np.array(eigenvalues)

# Load the data from the file
with open(f'datasets/Eigenvectors_results_{size_of_the_chain}_feast.dat', 'r') as f:
    lines = f.readlines()

# Split the lines into eigenvectors
eigenvectors = []
temp_vector = []
for line in lines[1:]:
    if line.strip():  # if line is not empty
        temp_vector.extend(line.strip().split())
    else:  # if line is empty
        eigenvectors.append(temp_vector)
        temp_vector = []

# Don't forget to add the last eigenvector
if temp_vector:
    eigenvectors.append(temp_vector)

# Convert the eigenvectors to a numpy array and reshape them
vectors = np.array(eigenvectors, dtype=float)    

print("This is the eigenvectors: ", vectors.shape, vectors[0].shape)
print("This is the eigenvalues: ", eigenvalues.shape)

#print(vectors[0].shape)
#print(vectors[:,0].shape)

S = 1/2 #spin number 
N = size_of_the_chain #size of the system

H = entropy_calc.Heisenberg(N, S)
print("Start the entropy calculation")

basis_H, basis_H_s_z, spins = H.calculate_basis()

#print("Basis H: ", basis_H)
#print("Basis S_z: ", basis_H_s_z)
print("List of possible values of S_z: ", spins)

all_energies = []
entropy_all_system = []
eigen_rho_sys_all = []
s_z_number = []
s_z_lambdas = []
sum_lambdas = []
psi_shape = []

#n_of_sites = int(len(adjMatrix)/2)
size_of_sub_A = int(size_of_the_chain/2)
size_of_sub_B = N - size_of_sub_A

print(f"This is size A {size_of_sub_A}")
print(f"This is size B {size_of_sub_B}")


#for i , spin in enumerate(spins):
i = int(size_of_the_chain/2) #spin number
    
spin_basis = H.sz_subspace(i)

print("S_z = " + str(spins[i]) + " start")
#print(f"Spin basis: {spin_basis}")

#calculation of new basis
subsystem_A, subsystem_B, new_basis = H.subsystems_fixed_s_z(spin_basis,size_of_sub_A,size_of_sub_B)

sum_entropy = 0
for j in range(len(eigenvalues)):

    psi = np.zeros(shape=(len(subsystem_A),len(subsystem_B)), dtype = float) #should deal only with floats
    
    for k,v in enumerate(vectors[j]):
        #print(f"This is {v}")
        psi[new_basis[k][0]][new_basis[k][1]] = v #0 and 1 becasue it's a matrix 
        #print("This is value of psi ", v)
        
    psi_shape.append(psi.shape)
    
    
    #print(f"This is psi vector {j}: \n", psi)
    #subsystemA
    rho = np.dot(psi, psi.conj().transpose()) 
    #print(f"This len is rho vector {j}: \n", len(rho))
    
    #trace calculation
    if not .9999999999999 <= np.trace(rho) <= 1.000000000001:
            print("Trace of the system: ", np.trace(rho))
    
    #entropy
    entropy_sys, eigen_rho_sys = H.calculate_entropy(rho, size_of_sub_A) #change to saving the lambdas
    #print(f"This is entropy of j-th {j} energy {energies[j]} with spin S_z {spins[i]} : {entropy_sys}")
    sum_entropy += entropy_sys
    
    #print(f"Lambdas : {eigen_rho_sys}")
    
    #[print(" lambdas: ", eigen_rho_sys[i]) for i  in range(len(eigen_rho_sys))]
    #[print("Complex lambdas: ", eigen_rho_sys[i]) for i  in range(len(eigen_rho_sys)) if eigen_rho_sys[i].imag <= 10e-8]
    
    entropy_all_system.append(entropy_sys)
    s_z_number.append(spins[i])
    
    s_z_lambdas.append([spins[i]]*len(eigen_rho_sys))
    #for lambdas from reduced density matrices
    eigen_rho_sys_all.append(eigen_rho_sys)

#psi_shape.append(psi.shape)    
#print(f"This is sum of this S_z {spins[i]} entropies {sum_entropy}")

#all_energies.append(eigenvalues)

for j in range(len(eigen_rho_sys_all)):
        sum_lambdas.append(sum(eigen_rho_sys_all[j]))

        
#all_energies = np.concatenate(all_energies)


print("Success Entropy")

print(eigenvalues)
print(entropy_all_system)
print(sum_lambdas)

# Create a DataFrame from your data
df = pd.DataFrame({
    'Eigenvalues': eigenvalues,
    'Entropy': entropy_all_system,
    'Sum_Lambdas': sum_lambdas
})

# Save the DataFrame to a CSV file
df.to_csv(f'{size_of_the_chain}_entropy_{int(spins[i])}_sz.csv', index=False)


print("Success Entropy save to file")
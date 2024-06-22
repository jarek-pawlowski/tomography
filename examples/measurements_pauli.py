import numpy as np
import functools as ft
import matplotlib.pyplot as plt
from scipy.stats import linregress

import src.utils_measure as utils


def measure_shadow(T,number_qubits):
    
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
        
    return snapshots

def reconstruct_from_shadow(T, rho0, snapshots, space_size, number_qubits):
    
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
#number_qubits = [2, 3]
#number_qubits = [2, 4, 6, 8, 10, 12]
number_qubits = [2, 4, 6, 8, 10, 12]
#set_of_T = [10, 100, 1000, 10000, 100000]
#set_of_T = [20, 50, 100, 150, 200]
set_of_T = [2, 3, 5, 10, 20, 50, 100, 150, 200]

all_data = []
for qubit in number_qubits:
    print(f"This is for {qubit}")
    
    space_size = int(pow(2, qubit))
    measurement = utils.Measurement(utils.Pauli, qubit, basis_c=utils.Pauli_c)
    
    #Pure state
    #psi_in = np.ones(tuple(2 for _ in range(qubit)), dtype=complex)/np.sqrt(2.**qubit) #reshape to 2x2 tensor (each 2x2 matrix describes a single qubit)
    
    #test for a Bell state: 
    #psi_in = np.zeros(tuple(2 for _ in range(qubit)), dtype=complex)
    #psi_in[(0,) * (qubit)] = 1./np.sqrt(2.)
    #psi_in[(1,) * (qubit)] = 1./np.sqrt(2.)
    
    #loaded states by user:
    psi_in = np.load(f'./training_states/training_states_{qubit}/{qubit}_tensor_state_1.npy')
    psi_in = psi_in.astype(np.complex128)

    rho0 = utils.tensordot(psi_in, psi_in, indices=0, conj_tr=(True,True)) #changed first to TRUE
    print(f"rho0 shape: {rho0.shape}")
    #print(f"rho0: {rho0}")
    norms = []
    mse = []
    for t in set_of_T: 
        print(f"Start {t}")
        snapshots = measure_shadow(t, qubit)
        rho1 = reconstruct_from_shadow(t, rho0, snapshots, space_size, qubit)
        norms.append(np.linalg.norm(rho0.reshape(space_size,space_size).real-rho1.reshape(space_size,space_size)))
        mse.append(np.mean((rho0.reshape(space_size, space_size).real - rho1.reshape(space_size, space_size).real)**2))
    
    
    #np.set_printoptions(linewidth=200)
    #print("rho initial: ")
    #print(np.around(rho0.reshape(space_size,space_size), 3))
    #print("\n")
    #print("rho reconstructed: ")
    #print(np.around(rho1, 3))
        

    fig, ax = plt.subplots()
    ax.plot(set_of_T, norms, '-o', label = f"{qubit} qubits")
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Setting the title
    ax.set_title(f"{qubit} qubits chain ground state")
    # Saving the figure
    #fig.savefig(f'./plots/{qubit}_qubits_shadow_real_chain.png')
    
    # Accumulate data for final plot
    all_data.append((set_of_T, norms, f"{qubit} qubits"))
    
    # Create additional plot to show all lines together
    fig, ax = plt.subplots()
    
    # Save x, y data to file
    #data = np.array([set_of_T, norms]).T
    #np.savetxt(f'{qubit}_data_to_compare.txt', data, delimiter=',', header='x, y', fmt='%.2f')
    data_mse = np.array([set_of_T, mse]).T
    np.savetxt(f'{qubit}_data_to_compare.txt', data_mse, delimiter=',', header='x, y', fmt='%.2f')
   
  
    '''
    # Create additional plot to show all lines together
    fig, ax = plt.subplots()
    
    for set_of_T, norms, label in all_data:
        # Plot the data
        ax.plot(set_of_T, norms, '-o', label=label)
        
        # Calculate the slope in log-log space
        log_set_of_T = np.log10(set_of_T)
        log_norms = np.log10(norms)
        slope, intercept, r_value, p_value, std_err = linregress(log_set_of_T, log_norms)
        
        # Annotate the slope on the plot
        x_position = set_of_T[len(set_of_T)//2]  # A middle x-value for positioning the annotation
        y_position = 10**(intercept + slope*np.log10(x_position))
        ax.annotate(f'{slope:.2f}', xy=(x_position, y_position), textcoords='offset points', xytext=(0,10), ha='center')
   
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("All qubits together Real chain log log")
    ax.legend()  # Display a legend to identify each line
    #fig.savefig('./plots/all_qubits_together_Real_chain_slope.png')
    fig.savefig('./plots/test.png') 
    '''
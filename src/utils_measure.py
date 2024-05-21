import numpy as np
from numpy.linalg import inv

#import QuantumTomography as qLib  # pip install Quantum-Tomography

def tensordot(a, b, indices=(1,0), moveaxis=None, conj_tr=(False,False)):
    a1 = np.conjugate(a.T) if conj_tr[0] else a  # warning: transposing reverses tensor indices
    b1 = np.conjugate(b.T) if conj_tr[1] else b  # warning: transposing reverses tensor indices
    result = np.tensordot(a1, b1, indices)
    if moveaxis is not None:
        result = np.moveaxis(result, *moveaxis)
    return result

def trace(a):
    # performs tensor contraction Tijk...ijk...
    dim = int(len(a.shape)/2)
    indices_to_sum = np.tile(np.indices([2]*dim).reshape(dim,-1).T, 2)
    return np.sum([a[tuple(idx)] for idx in indices_to_sum])

Pauli0 = np.array([[1,0],[0,1]])
PauliX = np.array([[0,1],[1,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1,0],[0,-1]])
Pauli_vector = [PauliX, PauliY, PauliZ]

G1 = np.kron(Pauli0,PauliX)/2
G2 = np.kron(Pauli0,PauliY)/2
G3 = np.kron(Pauli0,PauliZ)/2
G4 = np.kron(PauliX,Pauli0)/2
G5 = np.kron(PauliX,PauliX)/2
G6 = np.kron(PauliX,PauliY)/2
G7 = np.kron(PauliX,PauliZ)/2
G8 = np.kron(PauliY,Pauli0)/2
G9 = np.kron(PauliY,PauliX)/2
G10 = np.kron(PauliY,PauliY)/2
G11 = np.kron(PauliY,PauliZ)/2
G12 = np.kron(PauliZ,Pauli0)/2
G13 = np.kron(PauliZ,PauliX)/2
G14 = np.kron(PauliZ,PauliY)/2
G15 = np.kron(PauliZ,PauliZ)/2
G16 = np.kron(Pauli0,Pauli0)/2

class Tomography:
    # stuff for tomography (PHYSICAL REVIEW A, VOLUME 64, 052312)
    def __init__(self, no_qubits, single_qubit_projectors): 
        self.no_qubits = no_qubits
        self.rho_dim = np.power(2, self.no_qubits)
        self.tomo_dim = self.rho_dim**2
        # create two qubit projection basis 
        projection_basis = []
        for psi_i in single_qubit_projectors.basis:
            for psi_j in single_qubit_projectors.basis: 
                projection_basis.append(np.kron(psi_i, psi_j))
        self.projection_basis = projection_basis
        self.Gamma = np.array([G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16])
        assert self.tomo_dim == len(self.Gamma), "dim is differen than number of Gamma matrices"

    def calulate_B_inv(self):
        # determine inverse of B matrix (Eq. 3.12 in PHYSICAL REVIEW A, VOLUME 64, 052312)
        B = np.zeros((self.tomo_dim, self.tomo_dim), dtype=complex)
        for nu in range(self.tomo_dim):
            for mu in range(self.tomo_dim):
                B[nu,mu] = tensordot(self.projection_basis[nu], tensordot(self.Gamma[mu], self.projection_basis[nu]), conj_tr=(True,False)).item()
        self.B_inv = inv(B) 
        
    def reconstruct(self, measurements, enforce_positiv_sem=False):
        # calculate state reconstruction (Eq. 3.13 in PHYSICAL REVIEW A, VOLUME 64, 052312)
        r = np.matmul(self.B_inv, measurements)
        r = tensordot(self.Gamma, r, indices=(0,0))
        if enforce_positiv_sem:
            eigs = np.amin(np.linalg.eigvalsh(r))
            if eigs < 0.: r -= np.eye(self.rho_dim)*eigs            
        return r


class States:
    def __init__(self, Kwiat_Pauli_Basis): 
        if Kwiat_Pauli_Basis:
            # convention leading to standard Pauli matrices in (2.3) in PHYSICAL REVIEW A 64 052312
            self.R = np.array([[1],[0]])
            self.L = np.array([[0],[1]])
            self.H = np.array([[1],[1]])/np.sqrt(2)
            self.V = np.array([[1],[-1]])/np.sqrt(2)*1j
            self.D = np.array([[1+1j],[1-1j]])/2
        else:
            # Kwiat convention (PHYSICAL REVIEW A 64 052312)
            self.H = np.array([[1],[0]])
            self.V = np.array([[0],[1]])
            self.L = np.array([[1],[1j]])/np.sqrt(2)
            self.R = np.array([[1],[-1j]])/np.sqrt(2)
            self.D = np.array([[1],[1]])/np.sqrt(2)
            self.Db = np.array([[1],[-1]])/np.sqrt(2)
    
    
class Basis:
    def __init__(self, basis):
        self.basis = basis
        
    def __str__(self):
        return '\n'.join(str(matrix) for matrix in self.basis)
            
states = States(Kwiat_Pauli_Basis=False)

Kwiat = Basis([tensordot(states.H, states.H, conj_tr=(False,True)), 
               tensordot(states.V, states.V, conj_tr=(False,True)),
               tensordot(states.D, states.D, conj_tr=(False,True)),
               tensordot(states.R, states.R, conj_tr=(False,True))])
Pauli = Basis([tensordot(states.D, states.D, conj_tr=(False,True)),
               tensordot(states.L, states.L, conj_tr=(False,True)),
               tensordot(states.H, states.H, conj_tr=(False,True))])
Pauli_c = Basis([tensordot(states.Db, states.Db, conj_tr=(False,True)),
                 tensordot(states.R, states.R, conj_tr=(False,True)),
                 tensordot(states.V, states.V, conj_tr=(False,True))])

Kwiat_projectors = Basis([states.H, states.V, states.D, states.R])

#print each object in Pauli as a matrix: 
#print(Pauli)
#print(Pauli_c)


class Measurement:
    # nice compendium: https://arxiv.org/pdf/2201.07968.pdf
    def __init__(self, basis, no_qubits, basis_c=None):
        self.basis = basis.basis
        if basis_c is not None:
            self.basis_c = basis_c.basis
        self.no_qubits = no_qubits

    def measure_single(self, rho, qubit_index, basis_index, return_state=False):
        # measure single qubit using given operator
        Prho = rho.copy()
        Prho = tensordot(self.basis[basis_index], Prho, indices=(1,qubit_index), moveaxis=(0,qubit_index))
        prob = trace(Prho).real
        if return_state:
            Prho = tensordot(Prho, self.basis[basis_index], indices=(self.no_qubits+qubit_index,0), moveaxis=(-1,qubit_index))
            return prob, Prho/prob
        else:
            return prob
        
        
    def measure_single_pure(self, psi, qubit_index, basis_index, return_state=False):
         # measure single qubit in pure state using given operator 
        Ppsi = psi.copy()
        Ppsi = tensordot(self.basis[basis_index], Ppsi, indices=(1,qubit_index), moveaxis=(0,qubit_index))
        to_contract = tuple(np.arange(self.no_qubits))
        prob = tensordot(psi, Ppsi, indices=(to_contract[::-1], to_contract), conj_tr=(True,False)).item().real
        # to_contract[::-1] because transposing reverses tensor indices
        if return_state:
            random = np.random.random()
            if prob > random: 
                return 1, Ppsi/np.sqrt(prob)
            else:
                Ppsi = psi.copy()
                Ppsi = tensordot(self.basis_c[basis_index], Ppsi, indices=(1,qubit_index), moveaxis=(0,qubit_index))
                to_contract = tuple(np.arange(self.no_qubits))
                prob = tensordot(psi, Ppsi, indices=(to_contract[::-1], to_contract), conj_tr=(True,False)).item().real
                return -1, Ppsi/np.sqrt(prob)
        else:
            return prob

    def measure(self, rho, basis_indices, return_state=False):
        # measure all qubits using list of operators
        # basis_indices = which operator to use when measuring subsequent qubits
        Prho = rho.copy()
        for i, bi in enumerate(basis_indices):
            Prho = tensordot(self.basis[bi], Prho, indices=(1,i), moveaxis=(0,i))
        prob = trace(Prho).real
        if return_state:
            for i, bi in enumerate(basis_indices):
                Prho = tensordot(Prho, self.basis[bi], indices=(self.no_qubits+i,0), moveaxis=(-1,i))
            return prob, Prho/prob
        else:
            return prob
        
    def measure_pure(self, psi, basis_indices, return_state=False):
        # measure all qubits in pure state using list of operators
        # basis_indices = which operator to use when measuring subsequent qubits
        Ppsi = psi.copy()
        for i, bi in enumerate(basis_indices):
            Ppsi = tensordot(self.basis[bi], Ppsi, indices=(1,i), moveaxis=(0,i))
        to_contract = tuple(np.arange(self.no_qubits))
        prob = tensordot(psi, Ppsi, indices=(to_contract[::-1], to_contract), conj_tr=(True,False)).item().real
        # to_contract[::-1] because transposing reverses tensor indices
        if return_state:
            return prob, Ppsi/np.sqrt(prob)
        else:
            return prob


basis_for_Kwiat_code = np.array([np.concatenate((Kwiat_projectors.basis[i].flatten(), Kwiat_projectors.basis[j].flatten())) for i in [0,1,2,3] for j in [0,1,2,3]])        
        
class Kwiat_library:
    # compendium: https://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf    
    def __init__(self, basis):
        self.basis = basis
        # Initiate tomography object
        self.tomolib = qLib.Tomography()

    def run_tomography(self, measurements, intensity=None, method='MLE'):  # possible methods are: 'MLE', 'HMLE', 'LINEAR'
        if intensity is None:
            intensity = np.ones(len(self.basis))
        input = []
        for b, m, in zip(self.basis, measurements):
            input.append([1., 0, 0, m, *b])
        [rho_approx, intensity, fval] = self.tomolib.state_tomography(np.array(input), intensity, method=method)
        return rho_approx

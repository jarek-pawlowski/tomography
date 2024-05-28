import typing as t
import functools as ft

import torch
import torch.nn as nn

from src.torch_measure import Measurement
from src.torch_utils import tensordot


class LSTMMeasurementSelector(nn.Module):
    def __init__(self, num_qubits: int, 
                 basis_matrices: t.List[torch.Tensor], 
                 basis_matrices_c: t.List[torch.Tensor],
                 basis_reconstruction: t.List[torch.Tensor], 
                 layers: int = 2, hidden_size: int = 16,
                 max_num_snapshots: int = 16, 
                 device: str = torch.device('cpu')):
        super(LSTMMeasurementSelector, self).__init__()
        self.device = device
        self.max_num_snapshots = max_num_snapshots
        self.num_qubits = num_qubits
        self.basis_matrices = torch.stack(basis_matrices, dim=0).to(self.device) # shape (len(bases), 2, 2)
        self.basis_matrices_c = torch.stack(basis_matrices_c, dim=0).to(self.device) # shape (len(bases), 2, 2)
        self.basis_reconstruction = torch.stack(basis_reconstruction, dim=0).to(self.device) # shape (len(bases), 2, 2)
        self.basis_size = len(basis_matrices)
        self.basis_dim = self.num_qubits*self.basis_size
        self.dens_matrix_dim = 2 * (4 ** num_qubits)
        # self.measurement_predictor = MLP(layers, 1 + self.basis_dim, hidden_size, self.basis_sufficient_params)
        self.basis_selector = nn.LSTMCell(self.num_qubits + self.basis_dim, hidden_size)
        self.projectors = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, self.basis_size),
            nn.Softmax(dim=-1)
            #nn.PReLU()
        ) for _ in range(num_qubits)])
        self.measurement = Measurement(self.num_qubits)
        #self.matrix_reconstructor = LSTMDensityMatrixReconstructor(self.num_qubits + self.basis_dim, num_qubits, layers, hidden_size)  # <- here

    def take_snapshot(self, rho_k, measurement_basis_probability_k):
        # basis matrices now are not RANDOM !!! Now LSTM chooses 
        
        # take a snapshot
        snapshot = []
        p = rho_k.clone()
        #print(p)
        #print(f"this is a1 a2 a3 {measurement_basis_probability_k}")
        for i in range(self.num_qubits):
            
            a1 = measurement_basis_probability_k[i][0]
            a2 = measurement_basis_probability_k[i][1]
            a3 = measurement_basis_probability_k[i][2]
            
            #print(f"this is a1 a2 a3 {a1} {a2} {a3}")
            #normalised eigenvectoris from a1, a2, a3 probabilities from LSTM 
            # https://en.wikipedia.org/wiki/Pauli_matrices#Pauli_vectors
            a_vector_norm = torch.norm(torch.tensor([a1, a2, a3], dtype=torch.complex64)) #this must be 1! 
            
            norm_psi = 1/torch.sqrt((2*a_vector_norm*(a3+a_vector_norm)))
    
            psi_plus = norm_psi * torch.tensor([a3 + a_vector_norm, a1 + 1j * a2], dtype=torch.complex64).to(self.device)
            psi_minus = norm_psi * torch.tensor([1j*a2 - a1, a3 + a_vector_norm], dtype=torch.complex64).to(self.device)
            
            #print(f"this is norm: {a_vector_norm}")
            #print(f"psi plus {psi_plus}")
            #print(f"psi minus {psi_minus}")            
            
            #construction of measurment projector from eigenvectors and complementary one
            basis_matrix = tensordot(psi_plus, psi_plus, indices=([], []), conj_tr=(False,True))  # shape (num_qubits, 2, 2)
            basis_matrix_c = tensordot(psi_minus, psi_minus, indices=([], []), conj_tr=(False,True))
            
            #print(f"this is basis_matrix {basis_matrix}")
            #print(f'this is basis_matrix_c {basis_matrix_c}')
            
            #print(p)
            #print(basis_matrix)
            #print(basis_matrix_c)
            
            s, p = self.measurement.measure_single_pure(p, i, basis_matrix, basis_matrix_c, return_state=True)
            snapshot.append(s)
            
        return snapshot

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor):
        # model gets last basis and measurements (snapshot) as an input -- to be considered later
        snapshot, basis_comp_vector = first_measurement
        basis_predictor_input = torch.cat((snapshot, basis_comp_vector.view(-1, self.basis_dim)), dim=-1)
        
        h_i = torch.randn((snapshot.shape[0], self.basis_selector.hidden_size), device=self.device)
        c_i = torch.randn((snapshot.shape[0], self.basis_selector.hidden_size), device=self.device)
        # initial guesses:
        snapshots_with_basis_vector = [basis_predictor_input]
        basis_vectors = [basis_comp_vector]
        
        for i in range(self.max_num_snapshots - 1):  # in our case max_num_measurements = the number of snapshots
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            h_i, c_i  = self.basis_selector(basis_predictor_input, (h_i, c_i)) #LSTM cell to select basis
            # projector -> simple single-layer perceptron that predicts Pauli selections from LSTM's hidden representation
            measurement_basis_probability = torch.stack([projector(h_i) for projector in self.projectors], dim=1) # shape (batch, num_qubits, len(bases))
            #check if it is going to work better with sqrt
            measurement_basis_probability = torch.sqrt(measurement_basis_probability) #output of LSTM is power of 2 so normalise! 

            '''
            #random selection of basis matrices where a_i = 1 and respective ones are 0]
            #1 0 0 , 0 1 0, 0 0 1 working, now probabilties 1/2 1/2 0 in random permutation
            measurement_basis_probability_not_random = torch.zeros_like(measurement_basis_probability, requires_grad=True).clone()
        
            for i in range(measurement_basis_probability_not_random.shape[0]):
                for j in range(measurement_basis_probability_not_random.shape[1]):
                    random_index = torch.randperm(measurement_basis_probability_not_random.shape[2])[0]
                    measurement_basis_probability_not_random[i][j][random_index] = 1
                    
                    # Generate a permutation of the indices
                    #indices = torch.randperm(measurement_basis_probability_not_random.shape[2])
                    
                    # Select the first two indices and assign them a value of 0.5
                    #measurement_basis_probability_not_random[i][j][indices[0]] = 0.5
                    #measurement_basis_probability_not_random[i][j][indices[1]] = 0.5
                    
            measurement_basis_probability = measurement_basis_probability_not_random
            measurement_basis_probability = torch.sqrt(measurement_basis_probability)
            '''
            #print(measurement_basis_probability)
            
            # now make measurements
            snapshot_batch = []
            # iterate over batch (to be paralelized!)
            #print(f"this is rho {rho}")
            for rho_k, measurement_basis_probability_k in zip(rho, measurement_basis_probability):
                rho_k = torch.complex(rho_k[0], rho_k[1]).view(*[2]*self.num_qubits) #2 more
                #print(f"this is rho_k {rho_k}")
                snapshot_batch.append(self.take_snapshot(rho_k, measurement_basis_probability_k))
            snapshot_batch = torch.Tensor(snapshot_batch).to(snapshot.device)
            basis_predictor_input = torch.cat((snapshot_batch, measurement_basis_probability.view(-1, self.basis_dim)), dim=-1)
            snapshots_with_basis_vector.append(basis_predictor_input)
            basis_vectors.append(measurement_basis_probability)
            #sqrt on basis vectors
        # now make the reconstruction
        # unfold over snapshots
        #print(basis_vectors)
        rho_reconstructed = []
        for snapshot_with_basis in snapshots_with_basis_vector:
            rho_reconstructed_s = torch.zeros(rho.shape[0], rho.shape[-1]**self.num_qubits, rho.shape[-2]**self.num_qubits).to(torch.complex64).to(self.device) #changed to rho.shape[-1] form rho.shape[-2]
            #print(rho_reconstructed_s)
            measurement_shadow = snapshot_with_basis[:,:self.num_qubits]
            basis_vectors = snapshot_with_basis[:,self.num_qubits:].view(-1, self.num_qubits, self.basis_size)
            basis = torch.tensordot(basis_vectors.to(torch.complex64), self.basis_reconstruction, ([2],[0])) #combined pauli matrix for reconstruction
            rho_collection = []
            for iq in range(self.num_qubits):
                rho_collection.append((torch.eye(2).expand(basis.shape[0],-1,-1).to(self.device) + measurement_shadow[:,iq,None,None]*basis[:,iq,:,:]*3)/2.)  # (Eq. A5) in https://arxiv.org/pdf/2106.12627.pdf
            # don't know how to combine ft.reduce with batch
            # poor man's approach:
            
            rho_collection = torch.permute(torch.stack(rho_collection), (1,0,2,3))
            #print(f"This is {rho_collection}")
            # iterate over batch (to be paralelized!)
            for k, rho_collection_k in enumerate(rho_collection):
                #print(ft.reduce(torch.kron, rho_collection_k).shape)
                #print(rho_reconstructed_s[k].shape)
                rho_reconstructed_s[k] = ft.reduce(torch.kron, rho_collection_k)  # Kronecker product of (multiple) single-qubit rhos 
            rho_reconstructed.append(rho_reconstructed_s)
        # stack everything together:
        rho_reconstructed = torch.stack(rho_reconstructed).sum(dim=0)/self.max_num_snapshots
        # split each rho into real and imag parts:
        rho_reconstructed = torch.stack([rho_reconstructed.real, rho_reconstructed.imag], dim=1)
        # we return only the final reconstruction -- to be considered later
        #print(f" rho : {tensordot(rho[0][0], rho[0][0],indices=0, conj_tr=(True,True)).reshape(4,4)}")
        #print(f" rho reconstructed : {rho_reconstructed[0][0]}")
        
        return [rho_reconstructed], basis_vectors  # we also return predicted basis vectors to further regularize them

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class DensityMatrixReconstructor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16):
        super(DensityMatrixReconstructor, self).__init__()
        self.num_qubits = num_qubits
        self.mlp = MLP(layers, input_dim, hidden_size, 2 * (4 ** num_qubits))

    def forward(self, measurement_with_basis: torch.Tensor):
        return self.mlp(measurement_with_basis).view(-1, 2, 2 ** self.num_qubits, 2 ** self.num_qubits)
    

class LSTMDensityMatrixReconstructor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16):
        super(LSTMDensityMatrixReconstructor, self).__init__()
        self.num_qubits = num_qubits
        self.lstm = nn.LSTM(input_dim, hidden_size, layers, proj_size=2 * (4 ** num_qubits), batch_first=True)

    def forward(self, measurements_with_basis: torch.Tensor):
        out, _ = self.lstm(measurements_with_basis)
        return out.view(measurements_with_basis.shape[0], measurements_with_basis.shape[1], 2, 2 ** self.num_qubits, 2 ** self.num_qubits)
    
from copy import deepcopy
import typing as t
from itertools import product
from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTMDiscreteMeasurementSelectorNoMeasurements(nn.Module):
    def __init__(self, num_qubits: int , possible_basis_matrices: t.List[torch.Tensor], layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMDiscreteMeasurementSelectorNoMeasurements, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.bases = torch.stack(possible_basis_matrices, dim=0)
        self.multiqubit_bases_ids = list(product(range(len(self.bases)), repeat=self.num_qubits))
        self.measurement_selector = nn.LSTM(self.basis_dim*self.max_num_measurements, hidden_size, layers, proj_size=self.max_num_measurements, batch_first=True, bias=True)
        self.matrix_reconstructor = nn.LSTM(self.basis_dim + 1, hidden_size, layers, proj_size=2 * (4 ** num_qubits), batch_first=True, bias=True)

    def forward(self, all_measurements: t.List[t.Tuple[torch.Tensor, torch.Tensor]], rho: torch.Tensor, predict_target_basis: bool = False, add_noise_while_selecting_basis: bool = False, measurements_order: t.Optional[torch.Tensor] = None):
        measurements_with_basis = []
        basis_vectors = []
        for measurment_tuple in all_measurements:
            measurement, basis = measurment_tuple
            basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, self.num_qubits*2*2*2)
            reconstructor_input = torch.cat((measurement, basis_as_vector), dim=-1)
            basis_vectors.append(basis_as_vector)
            measurements_with_basis.append(reconstructor_input)

        measurements_with_basis = torch.stack(measurements_with_basis, dim=1) # shape (batch, max_num_measurements, num_qubits*2*2*2 + 1)

        basis_vectors = torch.stack(basis_vectors, dim=1).to(measurements_with_basis.device) # shape (batch, max_num_measurements, num_qubits*2*2*2)
        basis_vectors = basis_vectors.view(-1, 1, self.basis_dim*self.max_num_measurements).expand(-1, self.max_num_measurements, -1)
        proposed_basis_probabilities, _ = self.measurement_selector(basis_vectors)
        proposed_basis_probabilities = F.softmax(proposed_basis_probabilities, dim=-1)  # shape (batch, max_num_measurements, max_num_measurements)

        mask = torch.ones(proposed_basis_probabilities.shape[0], 1, proposed_basis_probabilities.shape[-1], device=proposed_basis_probabilities.device)
        mask[:, :, 0] = 0.
        current_basis_probabilities = proposed_basis_probabilities * mask
        proposed_basis_ids = [torch.zeros(measurements_with_basis.shape[0], dtype=torch.int64, device=measurements_with_basis.device)]
        current_basis_ids = torch.stack(proposed_basis_ids, dim=1)
        
        all_expected_basis_ids = deepcopy(proposed_basis_ids)
        for i in range(self.max_num_measurements - 1):
            if predict_target_basis:
                with torch.no_grad():
                    current_basis_ids = torch.stack(proposed_basis_ids, dim=1)  # shape (batch, max_num_measurements)
                    current_measurements_with_basis = torch.gather(measurements_with_basis, 1, current_basis_ids.unsqueeze(-1).expand(-1, -1, measurements_with_basis.shape[-1]))  # shape (batch, i, num_qubits*2*2*2 + 1)

                    current_measurements_with_basis_expanded = current_measurements_with_basis.view(-1, 1, current_measurements_with_basis.shape[1], self.basis_dim + 1).expand(-1, self.max_num_measurements, -1, -1)
                    current_measurements_with_target = torch.cat((current_measurements_with_basis_expanded, measurements_with_basis.unsqueeze(2)), dim=2)  # shape (batch, max_num_measurements, i + 1, num_qubits*2*2*2 + 1)
                    current_measurements_with_target_flatten = torch.flatten(current_measurements_with_target, start_dim=0, end_dim=1)  # shape (batch*max_num_measurements, i + 1, num_qubits*2*2*2 + 1)
                    current_reconstructions_flattened, _ = self.matrix_reconstructor(current_measurements_with_target_flatten)
                    current_reconstructions = torch.unflatten(current_reconstructions_flattened, dim=0, sizes=(current_measurements_with_target.shape[0], current_measurements_with_target.shape[1])) # shape (batch, max_num_measurements, i + 1, 2, 2**self.num_qubits, 2**self.num_qubits)

                    reconstruction_losses = torch.nn.functional.mse_loss(
                        current_reconstructions[:, :, -1].view(-1, self.max_num_measurements, 2, 2**self.num_qubits, 2**self.num_qubits),
                        rho.view(-1, 1, 2, 2**self.num_qubits, 2**self.num_qubits).expand(-1, self.max_num_measurements, -1, -1, -1),
                        reduction='none'
                    ).mean(dim=(-1, -2, -3))

                    if add_noise_while_selecting_basis:
                        noise = torch.randn_like(reconstruction_losses) * 1.e-4
                        reconstruction_losses = reconstruction_losses + noise

                    expected_basis_ids = torch.argmin(reconstruction_losses, dim=-1)  # shape (batch,)
            else:
                expected_basis_ids = torch.zeros(measurements_with_basis.shape[0], device=measurements_with_basis.device, dtype=torch.int64)

            proposed_basis_idx = torch.argmax(current_basis_probabilities[:, i], dim=-1)
            proposed_basis_ids.append(proposed_basis_idx)
            
            mask[torch.arange(mask.shape[0]), :, proposed_basis_idx] = 0.
            current_basis_probabilities = current_basis_probabilities * mask
            
            all_expected_basis_ids.append(expected_basis_ids)
        
        proposed_basis_ids = torch.stack(proposed_basis_ids, dim=1) # shape (batch, max_num_measurements)
        if measurements_order is not None:
            proposed_basis_ids = measurements_order.expand_as(proposed_basis_ids)

        proposed_measurements_with_basis = torch.gather(measurements_with_basis, 1, proposed_basis_ids.unsqueeze(-1).expand(-1, -1, measurements_with_basis.shape[-1]))  # shape (batch, max_num_measurements, num_qubits*2*2*2 + 1)
        reconstructed_matrices, _ = self.matrix_reconstructor(proposed_measurements_with_basis)
        reconstructed_matrices = reconstructed_matrices.view(-1, self.max_num_measurements, 2, 2**self.num_qubits, 2**self.num_qubits)
        all_expected_basis_ids = torch.stack(all_expected_basis_ids, dim=1)  # shape (batch, max_num_measurements)

        return reconstructed_matrices, proposed_basis_probabilities, all_expected_basis_ids

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


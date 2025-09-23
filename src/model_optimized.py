from copy import deepcopy
import typing as t
from itertools import product
from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tomography_utils_torch import measure, measure_batch_kron



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



class LSTMDiscreteMeasurementSelector(nn.Module):
    def __init__(self, num_qubits: int , possible_basis_matrices: t.List[torch.Tensor], layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMDiscreteMeasurementSelector, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.bases = torch.stack(possible_basis_matrices, dim=0)
        self.multiqubit_bases_ids = list(product(range(len(self.bases)), repeat=self.num_qubits))
        self.measurement_selector = nn.LSTMCell(1 + self.basis_dim, hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, self.max_num_measurements),
            nn.Softmax(dim=-1)
        )
        self.matrix_reconstructor = nn.LSTM(self.basis_dim + 1, hidden_size, layers, proj_size=2 * (4 ** num_qubits), batch_first=True, bias=True)

    def forward(self, all_measurements: t.List[t.Tuple[torch.Tensor, torch.Tensor]], rho: torch.Tensor, predict_target_basis: bool = False, add_noise_while_selecting_basis: bool = False, measurements_order: t.Optional[torch.Tensor] = None):
        measurements_with_basis = []
        for measurment_tuple in all_measurements:
            measurement, basis = measurment_tuple
            basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, self.num_qubits*2*2*2)
            reconstructor_input = torch.cat((measurement, basis_as_vector), dim=-1)
            measurements_with_basis.append(reconstructor_input)

        measurements_with_basis = torch.stack(measurements_with_basis, dim=1) # shape (batch, max_num_measurements, num_qubits*2*2*2 + 1)

        h_i = torch.randn((measurements_with_basis.shape[0], self.measurement_selector.hidden_size), device=measurements_with_basis.device)
        c_i = torch.randn((measurements_with_basis.shape[0], self.measurement_selector.hidden_size), device=measurements_with_basis.device)

        proposed_basis_ids = [torch.zeros(measurements_with_basis.shape[0], dtype=torch.int64, device=measurements_with_basis.device)]
        all_expected_basis_ids = deepcopy(proposed_basis_ids)
        mask = torch.ones(measurements_with_basis.shape[0], self.max_num_measurements, device=measurements_with_basis.device)

        init_proposed_basis_probabilities = torch.zeros(measurements_with_basis.shape[0], self.max_num_measurements, device=measurements_with_basis.device)
        init_proposed_basis_probabilities[:, 0] = 1.0
        all_predicted_basis_probabilities = [init_proposed_basis_probabilities]

        for i in range(self.max_num_measurements - 1):
            current_basis_ids = torch.stack(proposed_basis_ids, dim=1)
            current_measurements_with_basis = torch.gather(measurements_with_basis, 1, current_basis_ids.unsqueeze(-1).expand(-1, -1, measurements_with_basis.shape[-1]))  # shape (batch, i, num_qubits*2*2*2 + 1)
        
            if predict_target_basis:
                with torch.no_grad():
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

            all_expected_basis_ids.append(expected_basis_ids)


            last_measurement_with_basis = current_measurements_with_basis[:, -1]  # shape (batch, num_qubits*2*2*2 + 1)
            h_i, c_i  = self.measurement_selector(last_measurement_with_basis, (h_i, c_i))
            proposed_basis_probabilities = self.projector(h_i) # shape (batch, max_num_measurements)

            mask[torch.arange(mask.shape[0]), proposed_basis_ids[-1]] = 0.
            current_basis_probabilities = proposed_basis_probabilities * mask

            proposed_basis_idx = torch.argmax(current_basis_probabilities, dim=-1)
            proposed_basis_ids.append(proposed_basis_idx)
            all_predicted_basis_probabilities.append(proposed_basis_probabilities)
            
        
        proposed_basis_ids = torch.stack(proposed_basis_ids, dim=1) # shape (batch, max_num_measurements)
        if measurements_order is not None:
            proposed_basis_ids = measurements_order.expand_as(proposed_basis_ids)

        proposed_measurements_with_basis = torch.gather(measurements_with_basis, 1, proposed_basis_ids.unsqueeze(-1).expand(-1, -1, measurements_with_basis.shape[-1]))  # shape (batch, max_num_measurements, num_qubits*2*2*2 + 1)
        reconstructed_matrices, _ = self.matrix_reconstructor(proposed_measurements_with_basis)
        reconstructed_matrices = reconstructed_matrices.view(-1, self.max_num_measurements, 2, 2**self.num_qubits, 2**self.num_qubits)
        all_expected_basis_ids = torch.stack(all_expected_basis_ids, dim=1)  # shape (batch, max_num_measurements)
        all_predicted_basis_probabilities = torch.stack(all_predicted_basis_probabilities, dim=1)  # shape (batch, max_num_measurements, max_num_measurements)

        return reconstructed_matrices, all_predicted_basis_probabilities, all_expected_basis_ids

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))




class LSTMCELLDiscreteMeasurementSelector(nn.Module):
    def __init__(self, num_qubits: int , hidden_size: int = 16, max_num_measurements: int = 16, selection_measurements: bool = False, temperature: float = 1.0, num_layers: int = 1):
        super(LSTMCELLDiscreteMeasurementSelector, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        # self.rho_dim = 2 * (4 ** num_qubits)
        self.measurements_dim = 4 ** num_qubits
        self.num_layers = num_layers

        self.selection_measurements = selection_measurements
        self.selector_input_dim = self.basis_dim + 1 if selection_measurements else self.basis_dim
        self.measurement_selector = nn.ModuleList(
            [nn.LSTMCell(self.selector_input_dim, hidden_size)] +
            [nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.selector_projector = nn.Sequential(
            nn.Linear(hidden_size, self.max_num_measurements),
            nn.Softmax(dim=-1)
        )
        # Measure memory
        self.matrix_reconstructor = nn.ModuleList(
            [nn.LSTMCell(self.basis_dim + self.max_num_measurements, hidden_size)] +
            [nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.reconstructor_projector = nn.Linear(hidden_size, 2 * (4 ** num_qubits))
        self.T = temperature

    def selector_forward(
        self,
        measurement_id: torch.Tensor,
        measurements: torch.Tensor,
        bases: torch.Tensor,
        selector_states: t.List[t.Tuple[torch.Tensor, torch.Tensor]],
        used_measurements_mask: torch.Tensor,
        measurements_memory: torch.Tensor,
    ):
        current_basis = bases[torch.arange(bases.shape[0]), measurement_id]

        if self.selection_measurements:
            selector_input = torch.cat((measurements_memory, current_basis), dim=-1)
        else:
            selector_input = current_basis

        multilayer_selector_states = []
        for i in range(self.num_layers):
            new_selector_states = self.measurement_selector[i](selector_input, selector_states[i])
            selector_input = new_selector_states[0]
            multilayer_selector_states.append(new_selector_states)

        proposed_basis_probabilities = self.selector_projector(multilayer_selector_states[-1][0]) # shape (batch, max_num_measurements)
        proposed_basis_probabilities = F.softmax(proposed_basis_probabilities / self.T, dim=-1)
        masked_probabilities = proposed_basis_probabilities * used_measurements_mask
        renormalized_masked_probabilities = masked_probabilities / (masked_probabilities.sum(dim=-1, keepdim=True) + 1e-8)
        proposed_measurement_id = torch.argmax(renormalized_masked_probabilities, dim=-1) # shape (batch,)

        measurements_memory = measurements_memory.clone()
        measurements_memory[torch.arange(measurements.shape[0]), proposed_measurement_id] = measurements[torch.arange(measurements.shape[0]), proposed_measurement_id]
        return proposed_measurement_id, renormalized_masked_probabilities, multilayer_selector_states, measurements_memory
    

    def reconstructor_forward(
        self,
        proposed_measurement_probabilities: torch.Tensor,
        bases: torch.Tensor,
        reconstructor_states: t.List[t.Tuple[torch.Tensor, torch.Tensor]],
        measurements_memory: torch.Tensor,
        # global_memory: torch.Tensor,
    ):
        # measure_basis_memory = (measurements_memory.unsqueeze(-1) * bases).sum(dim=-2) # shape (batch, num_qubits*2*2*2)

        soft_basis = (proposed_measurement_probabilities.unsqueeze(-1) * bases).sum(dim=-2) # shape (batch, num_qubits*2*2*2)
        # memory_squeezed = global_memory.view(global_memory.shape[0], -1)  # shape (batch, rho_dim)
        reconstructor_input = torch.cat((measurements_memory, soft_basis), dim=-1)

        multilayer_reconstructor_states = []
        for i in range(self.num_layers):
            new_reconstructor_states = self.matrix_reconstructor[i](reconstructor_input, reconstructor_states[i])
            reconstructor_input = new_reconstructor_states[0]
            multilayer_reconstructor_states.append(new_reconstructor_states)

        predicted_rho = self.reconstructor_projector(multilayer_reconstructor_states[-1][0]).view(-1, 2, 2**self.num_qubits, 2**self.num_qubits)

        return predicted_rho, multilayer_reconstructor_states
    
    def initialize_hidden_states(self, batch_size: int, device: torch.device):
        multilayer_selector_states = []
        multilayer_reconstructor_states = []
        for i in range(self.num_layers):
            selector_hidden_state = torch.zeros(batch_size, self.measurement_selector[i].hidden_size, device=device)
            selector_cell_state = torch.zeros(batch_size, self.measurement_selector[i].hidden_size, device=device)
            reconstructor_hidden_state = torch.zeros(batch_size, self.matrix_reconstructor[i].hidden_size, device=device)
            reconstructor_cell_state = torch.zeros(batch_size, self.matrix_reconstructor[i].hidden_size, device=device)
            multilayer_selector_states.append((selector_hidden_state, selector_cell_state))
            multilayer_reconstructor_states.append((reconstructor_hidden_state, reconstructor_cell_state))
        return multilayer_selector_states, multilayer_reconstructor_states

    def forward(
        self,
        measurement_id: torch.Tensor,
        measurements: torch.Tensor,
        bases: torch.Tensor,
        selector_states: t.Tuple[torch.Tensor, torch.Tensor],
        reconstructor_states: t.Tuple[torch.Tensor, torch.Tensor],
        used_measurements_mask: torch.Tensor,
        # global_memory: torch.Tensor,
        measurements_memory: torch.Tensor,
    ):
        '''
        Assumes:
            measurement_id: Tensor of shape (batch,)
            measurements: Tensor of shape (batch, max_num_measurements)
            bases: Tensor of shape (batch, max_num_measurements, num_qubits*2*2*2)
            selector_states: Tuple of (hidden_state, cell_state) for the selector LSTM
            reconstructor_states: Tuple of (hidden_state, cell_state) for the reconstructor LSTM
            used_measurements_mask: Tensor of shape (batch, max_num_measurements)
            global_memory: Tensor of shape (batch, 2, 2**num_qubits, 2**num_qubits)
            measurements_memory: Tensor of shape (batch, max_num_measurements)
        '''

        proposed_measurement_id, proposed_basis_probabilities, new_selector_states, measurements_memory = self.selector_forward(
            measurement_id,
            measurements,
            bases,
            selector_states,
            used_measurements_mask,
            measurements_memory
        )

        predicted_rho, new_reconstructor_states = self.reconstructor_forward(
            proposed_basis_probabilities,
            bases,
            reconstructor_states,
            measurements_memory
        )

        return proposed_measurement_id, predicted_rho, new_selector_states, new_reconstructor_states, measurements_memory

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))



class CombinedLSTMDiscreteMeasurementSelector(nn.Module):
    def __init__(self, num_qubits: int , hidden_size: int = 16, max_num_measurements: int = 16, selection_measurements: bool = False, temperature: float = 1.0, num_layers: int = 1):
        super().__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.lstm_cell = LSTMCELLDiscreteMeasurementSelector(num_qubits, hidden_size, max_num_measurements, selection_measurements, temperature, num_layers)
        # self.memory_selector = nn.Conv2d(4, 2, kernel_size=1)

    def forward(
        self,
        measurements: torch.Tensor,
        bases: torch.Tensor,
        rho: torch.Tensor,
        first_measurement_id: int = 0,
        criterions: t.Dict[str, t.Callable] = {}
    ):
        selector_states, init_reconstructor_states = self.lstm_cell.initialize_hidden_states(measurements.shape[0], measurements.device)

        init_measurement_probabilities = torch.zeros(measurements.shape[0], self.max_num_measurements, device=measurements.device)
        init_measurement_probabilities[:, first_measurement_id] = 1.0

        proposed_measurement_id = torch.full((measurements.shape[0],), first_measurement_id, dtype=torch.int64, device=measurements.device)

        measurements_memory = torch.zeros_like(measurements)
        measurements_memory[torch.arange(measurements.shape[0]), proposed_measurement_id] = measurements[torch.arange(measurements.shape[0]), proposed_measurement_id]

        # global_memory = torch.zeros_like(rho)

        predicted_rho, reconstructor_states = self.lstm_cell.reconstructor_forward(
            init_measurement_probabilities,
            bases,
            init_reconstructor_states,
            measurements_memory
            # global_memory
        )

        # global_memory = self.memory_selector(torch.cat((global_memory, predicted_rho), dim=1))

        measurements_mask = torch.ones_like(measurements)
        measurements_mask[:, first_measurement_id] = 0.

        loss = F.mse_loss(predicted_rho, rho)
        
        metrics = {name: {} for name in criterions.keys()}
        for name, criterion in criterions.items():
            metrics[name]['measurement 0'] = criterion(predicted_rho, rho)

        for i in range(self.max_num_measurements):
            proposed_measurement_id, predicted_rho, selector_states, reconstructor_states, measurements_memory = self.lstm_cell(
                proposed_measurement_id,
                measurements,
                bases,
                selector_states,
                reconstructor_states,
                measurements_mask,
                measurements_memory,
                # global_memory
            )

            measurements_mask = measurements_mask.clone()
            measurements_mask[torch.arange(measurements_mask.shape[0]), proposed_measurement_id] = 0.

            loss = loss + F.mse_loss(predicted_rho, rho)

            for name, criterion in criterions.items():
                metrics[name][f'measurement {i}'] = criterion(predicted_rho, rho)

        return loss, metrics

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))



class LSTMCELLMeasurementPredictor(nn.Module):
    def __init__(self, num_qubits: int , hidden_size: int = 16, max_num_measurements: int = 16, selection_measurements: bool = False, num_layers: int = 1):
        super(LSTMCELLMeasurementPredictor, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.basis_sufficient_params = num_qubits*4
        self.measurements_dim = 4 ** num_qubits
        self.num_layers = num_layers

        self.selection_measurements = selection_measurements
        self.selector_input_dim = self.basis_dim + self.max_num_measurements if selection_measurements else self.basis_dim
        self.measurement_predictor = nn.ModuleList(
            [nn.LSTMCell(self.selector_input_dim, hidden_size)] +
            [nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.predictor_projector = nn.Sequential(
            nn.Linear(hidden_size, self.basis_sufficient_params)
        )
        # Measure memory
        self.matrix_reconstructor = nn.ModuleList(
            [nn.LSTMCell(self.basis_dim + self.max_num_measurements, hidden_size)] +
            [nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.reconstructor_projector = nn.Sequential(
            nn.Linear(hidden_size, 2 * (4 ** num_qubits))
        )

    def predictor_forward(
        self,
        rho: torch.Tensor,
        basis: torch.Tensor,
        predictor_states: t.List[t.Tuple[torch.Tensor, torch.Tensor]],
        measurements_memory: torch.Tensor,
        measurement_idx: int,
    ):
        if self.selection_measurements:
            selector_input = torch.cat((measurements_memory, basis), dim=-1)
        else:
            selector_input = basis

        # new_predictor_states = self.measurement_predictor(selector_input, predictor_states)

        multilayer_predictor_states = []
        for i in range(self.num_layers):
            new_predictor_states = self.measurement_predictor[i](selector_input, predictor_states[i])
            selector_input = new_predictor_states[0]
            multilayer_predictor_states.append(new_predictor_states)

        proposed_basis_raw = self.predictor_projector(multilayer_predictor_states[-1][0]) # shape (batch, max_num_measurements)
        basis_matrices = self.construct_basis_matrices(proposed_basis_raw)
        
        new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(-1, self.num_qubits*2*2*2)
        
        rho_complex = torch.complex(rho[:, 0], rho[:, 1]).view(-1, 2**self.num_qubits, 2**self.num_qubits)
        new_measurements = measure_batch_kron(rho_complex, basis_matrices).view(-1, 1)

        measurements_memory = measurements_memory.clone()
        measurements_memory[torch.arange(new_measurements.shape[0]), measurement_idx] = new_measurements.squeeze(-1) # shape (batch, max_num_measurements)
            
        return new_basis_as_vector, measurements_memory, multilayer_predictor_states
    
    def construct_basis_matrices(self, measurement_basis_vectors: torch.Tensor):
        basis_vectors = measurement_basis_vectors.view(-1, self.num_qubits, 2, 2)
        basis_vectors = torch.complex(basis_vectors[..., 0], basis_vectors[..., 1])
        basis_matrices = torch.zeros(basis_vectors.shape[0], self.num_qubits, 2, 2, dtype=torch.complex64, device=measurement_basis_vectors.device)
        basis_matrices[..., 0, 0] = basis_vectors[..., 0].abs() ** 2
        basis_matrices[..., 1, 1] = basis_vectors[..., 1].abs() ** 2
        basis_matrices[..., 0, 1] = basis_vectors[..., 0] * basis_vectors[..., 1].conj()
        basis_matrices[..., 1, 0] = basis_vectors[..., 1] * basis_vectors[..., 0].conj() 
        basis_matrices /= torch.vmap(torch.vmap(torch.trace))(basis_matrices).view(*basis_matrices.shape[:-2], 1, 1)
        return basis_matrices


    def reconstructor_forward(
        self,
        bases: torch.Tensor,
        reconstructor_states: t.List[t.Tuple[torch.Tensor, torch.Tensor]],
        measurements_memory: torch.Tensor,
        # global_memory: torch.Tensor,
    ):
        reconstructor_input = torch.cat((measurements_memory, bases), dim=-1)

        # new_reconstructor_states = self.matrix_reconstructor(reconstructor_input, reconstructor_states)

        multilayer_reconstructor_states = []
        for i in range(self.num_layers):
            new_reconstructor_states = self.matrix_reconstructor[i](reconstructor_input, reconstructor_states[i])
            reconstructor_input = new_reconstructor_states[0]
            multilayer_reconstructor_states.append(new_reconstructor_states)

        predicted_rho = self.reconstructor_projector(multilayer_reconstructor_states[-1][0]).view(-1, 2, 2**self.num_qubits, 2**self.num_qubits)

        return predicted_rho, multilayer_reconstructor_states

    def initialize_hidden_states(self, batch_size: int, device: torch.device):
        multilayer_predictor_states = []
        multilayer_reconstructor_states = []
        for i in range(self.num_layers):
            # predictor_hidden_state = torch.zeros(batch_size, self.measurement_predictor[i].hidden_size, device=device)
            # predictor_cell_state = torch.zeros(batch_size, self.measurement_predictor[i].hidden_size, device=device)
            reconstructor_hidden_state = torch.zeros(batch_size, self.matrix_reconstructor[i].hidden_size, device=device)
            reconstructor_cell_state = torch.zeros(batch_size, self.matrix_reconstructor[i].hidden_size, device=device)
            predictor_hidden_state = torch.randn(batch_size, self.measurement_predictor[i].hidden_size, device=device)
            predictor_cell_state = torch.randn(batch_size, self.measurement_predictor[i].hidden_size, device=device)
            # reconstructor_hidden_state = torch.randn(batch_size, self.matrix_reconstructor[i].hidden_size, device=device)
            # reconstructor_cell_state = torch.randn(batch_size, self.matrix_reconstructor[i].hidden_size, device=device)
            multilayer_predictor_states.append((predictor_hidden_state, predictor_cell_state))
            multilayer_reconstructor_states.append((reconstructor_hidden_state, reconstructor_cell_state))
        return multilayer_predictor_states, multilayer_reconstructor_states

    def forward(
        self,
        rho: torch.Tensor,
        basis: torch.Tensor,
        predictor_states: t.List[t.Tuple[torch.Tensor, torch.Tensor]],
        reconstructor_states: t.List[t.Tuple[torch.Tensor, torch.Tensor]],
        measurements_memory: torch.Tensor,
        measurement_idx: int,
    ):
        '''
        Assumes:
            rho: Tensor of shape (batch, 2, 2**num_qubits, 2**num_qubits)
            basis: Tensor of shape (batch, num_qubits*2*2*2)
            predictor_states: Tuple of (hidden_state, cell_state) for the predictor LSTM
            reconstructor_states: Tuple of (hidden_state, cell_state) for the reconstructor LSTM
            measurements_memory: Tensor of shape (batch, max_num_measurements)
        '''

        proposed_basis, measurements_memory, new_predictor_states = self.predictor_forward(
            rho,
            basis,
            predictor_states,
            measurements_memory,
            measurement_idx
        )

        predicted_rho, new_reconstructor_states = self.reconstructor_forward(
            proposed_basis,
            reconstructor_states,
            measurements_memory,
        )

        return proposed_basis, predicted_rho, new_predictor_states, new_reconstructor_states, measurements_memory

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class CombinedLSTMMeasurementPredictor(nn.Module):
    def __init__(self, num_qubits: int , hidden_size: int = 16, max_num_measurements: int = 16, selection_measurements: bool = False, num_layers: int = 1, measurements_weights: bool = False):
        super().__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.lstm_cell = LSTMCELLMeasurementPredictor(num_qubits, hidden_size, max_num_measurements, selection_measurements, num_layers)
        self.measurements_weights = measurements_weights

    def forward(
        self,
        measurements: torch.Tensor,
        bases: torch.Tensor,
        rho: torch.Tensor,
        first_measurement_id: int = 0,
        criterions: t.Dict[str, t.Callable] = {},
        return_rhos: bool = False,
    ):
        predictor_states, init_reconstructor_states = self.lstm_cell.initialize_hidden_states(measurements.shape[0], measurements.device)

        proposed_measurement = measurements[:, first_measurement_id].unsqueeze(-1)
        proposed_basis = bases[:, first_measurement_id]

        measurements_memory = torch.zeros_like(measurements)
        measurements_memory[torch.arange(measurements.shape[0]), 0] = proposed_measurement.squeeze(-1) # shape (batch, max_num_measurements)

        predicted_rho, reconstructor_states = self.lstm_cell.reconstructor_forward(
            proposed_basis,
            init_reconstructor_states,
            measurements_memory,
        )

        if return_rhos:
            all_predicted_rhos = [predicted_rho]

        loss = F.mse_loss(predicted_rho, rho)
        
        metrics = {name: {} for name in criterions.keys()}
        for name, criterion in criterions.items():
            metrics[name]['measurement 0'] = criterion(predicted_rho, rho)

        for i in range(1, self.max_num_measurements):
            proposed_basis, predicted_rho, predictor_states, reconstructor_states, measurements_memory = self.lstm_cell(
                rho,
                proposed_basis,
                predictor_states,
                reconstructor_states,
                measurements_memory,
                measurement_idx=i,
            )

            if return_rhos:
                all_predicted_rhos.append(predicted_rho)

            weight = 1.
            if self.measurements_weights:
                weight = self.max_num_measurements / (i + 1)

            loss = loss + weight * F.mse_loss(predicted_rho, rho)

            for name, criterion in criterions.items():
                metrics[name][f'measurement {i}'] = criterion(predicted_rho, rho)

        if return_rhos:
            return loss, metrics, torch.stack(all_predicted_rhos, dim=1)
        return loss, metrics

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class LSTMReconstructor(nn.Module):
    def __init__(self, num_qubits: int, hidden_size: int = 16, max_num_measurements: int = 16, num_layers: int = 1):
        super(LSTMReconstructor, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.num_layers = num_layers
        self.matrix_reconstructor = nn.ModuleList(
            [nn.LSTMCell(self.basis_dim + self.max_num_measurements, hidden_size)] +
            [nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.reconstructor_projector = nn.Linear(hidden_size, 2 * (4 ** num_qubits))

    def forward(
        self,
        measurements: torch.Tensor,
        bases: torch.Tensor,
        rho: torch.Tensor,
        criterions: t.Dict[str, t.Callable] = {}
    ):
        reconstructor_states = self.initialize_hidden_states(measurements.shape[0], measurements.device)
        measurements_memory = torch.zeros_like(measurements)

        loss = 0.
        metrics = {name: {} for name in criterions.keys()}

        for i in range(self.max_num_measurements):
            measurements_memory = measurements_memory.clone()
            measurements_memory[torch.arange(measurements.shape[0]), i] = measurements[:, i].squeeze(-1) # shape (batch, max_num_measurements)
            current_basis = bases[:, i]

            reconstructor_input = torch.cat((measurements_memory, current_basis), dim=-1)

            multilayer_reconstructor_states = []
            for j in range(self.num_layers):
                new_reconstructor_states = self.matrix_reconstructor[j](reconstructor_input, reconstructor_states[j])
                reconstructor_input = new_reconstructor_states[0]
                multilayer_reconstructor_states.append(new_reconstructor_states)
            
            reconstructor_states = multilayer_reconstructor_states

            predicted_rho = self.reconstructor_projector(multilayer_reconstructor_states[-1][0]).view(-1, 2, 2**self.num_qubits, 2**self.num_qubits)

            loss = loss + F.mse_loss(predicted_rho, rho)

            for name, criterion in criterions.items():
                metrics[name][f'measurement {i}'] = criterion(predicted_rho, rho)

        return loss, metrics
    
    def initialize_hidden_states(self, batch_size: int, device: torch.device):
        multilayer_reconstructor_states = []
        for i in range(self.num_layers):
            reconstructor_hidden_state = torch.zeros(batch_size, self.matrix_reconstructor[i].hidden_size, device=device)
            reconstructor_cell_state = torch.zeros(batch_size, self.matrix_reconstructor[i].hidden_size, device=device)
            multilayer_reconstructor_states.append((reconstructor_hidden_state, reconstructor_cell_state))
        return multilayer_reconstructor_states

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
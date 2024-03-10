import typing as t

import torch
import torch.nn as nn

from src.torch_measure import measure


class MLP(nn.Module):
    def __init__(self, layers: int, input_dim: int, hidden_size: int, output_dim: int):
        super(MLP, self).__init__()
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_size))
        mlp.append(nn.ReLU())
        for i in range(layers - 1):
            mlp.append(nn.Linear(hidden_size, hidden_size))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(hidden_size, output_dim))
        self.mlp = nn.Sequential(*mlp)
    
    def forward(self, x: torch.Tensor):
        return self.mlp(x)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class Regressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, layers: int = 2, hidden_size: int = 128, input_dropout: float = 0.0):
        super(Regressor, self).__init__()
        self.dropout = nn.Dropout(input_dropout)
        self.mlp = self._get_mlp(layers, input_dim, hidden_size, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        return self.mlp(x)
    
    def _get_mlp(self, layers: int, input_dim: int, hidden_size: int, output_dim: int):
        mlp = []
        mlp.append(nn.Linear(input_dim, hidden_size))
        mlp.append(nn.ReLU())
        for i in range(layers - 1):
            mlp.append(nn.Linear(hidden_size, hidden_size))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(hidden_size, output_dim))
        return nn.Sequential(*mlp)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class Classifier(Regressor):
    def __init__(self, input_dim: int, output_dim: int = 1, layers: int = 2, hidden_size: int = 128, input_dropout: float = 0.0):
        super(Classifier, self).__init__(input_dim, output_dim, layers, hidden_size, input_dropout)
    
    def forward(self, x: torch.Tensor):
        x = super(Classifier, self).forward(x)
        return torch.sigmoid(x)


class SequentialMeasurementPredictor(nn.Module):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(SequentialMeasurementPredictor, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.measurement_predictors = nn.ModuleList([
            MLP(layers, i * (1 + num_qubits*2*4), hidden_size, num_qubits*4) for i in range(1, max_num_measurements + 1)
        ])
        self.matrix_reconstructors = nn.ModuleList([
            DensityMatrixReconstructor(i *(1 + num_qubits*2*4), num_qubits, layers, hidden_size) for i in range(1, max_num_measurements + 1)
        ])

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
        measurement_with_basis = torch.cat((measurement, basis_as_vector), dim=-1)

        reconstructed_matrices = []
        for i in range(self.max_num_measurements):
            reconstructed_matrix = self.matrix_reconstructors[i](measurement_with_basis)
            measurement_basis_vectors = self.measurement_predictors[i](measurement_with_basis)
            new_measurement_with_basis = []
            for rho_k, measurement_basis_vector in zip(rho, measurement_basis_vectors):
                rho_k = torch.complex(rho_k[0], rho_k[1]).view(*[2, 2]*self.num_qubits)
                basis_vectors = measurement_basis_vector.view(2, 2, 2)
                basis_vectors = torch.complex(basis_vectors[..., 0], basis_vectors[..., 1])

                basis_matrices = torch.zeros(2, 2, 2, dtype=torch.complex64, device=rho_k.device)
                basis_matrices[:, 0, 0] = basis_vectors[:, 0].abs() ** 2
                basis_matrices[:, 1, 1] = basis_vectors[:, 1].abs() ** 2
                basis_matrices[:, 0, 1] = basis_vectors[:, 0] * basis_vectors[:, 1].conj()
                basis_matrices[:, 1, 0] = basis_vectors[:, 1] * basis_vectors[:, 0].conj() 
                basis_matrices[0] /= basis_matrices[0].trace()
                basis_matrices[1] /= basis_matrices[1].trace()

                new_measurement = measure(rho_k, (basis_matrices[0], basis_matrices[1]))
                new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(basis.shape[1]*2*2*2)
                new_measurement_with_basis.append(torch.cat((new_measurement.unsqueeze(0), new_basis_as_vector), dim=-1))

            new_measurement_with_basis = torch.stack(new_measurement_with_basis, dim=0)
            measurement_with_basis = torch.cat([measurement_with_basis, new_measurement_with_basis], dim=1)
            reconstructed_matrices.append(reconstructed_matrix)
        reconstructed_matrices = torch.stack(reconstructed_matrices, dim=1)
        return reconstructed_matrices

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class RecurrentMeasurementPredictor(nn.Module):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(RecurrentMeasurementPredictor, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.basis_sufficient_params = num_qubits*4
        self.dens_matrix_dim = 2 * (4 ** num_qubits)
        self.measurement_predictor = MLP(layers, 1 + self.basis_dim, hidden_size, self.basis_sufficient_params)
        self.matrix_reconstructor = DensityMatrixReconstructor(1 + self.basis_dim + self.dens_matrix_dim, num_qubits, layers, hidden_size)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
        measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)

        reconstructed_matrix = torch.zeros((measurement.shape[0], self.dens_matrix_dim), device=measurement.device)

        reconstructed_matrices = []
        for i in range(self.max_num_measurements):
            reconstructor_input = torch.cat((measurement_predictor_input, reconstructed_matrix), dim=-1)
            reconstructed_matrix = self.matrix_reconstructor(reconstructor_input)
            measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            new_measurement_predictor_input = []
            for rho_k, measurement_basis_vector in zip(rho, measurement_basis_vectors):
                rho_k = torch.complex(rho_k[0], rho_k[1]).view(*[2, 2]*self.num_qubits)
                basis_vectors = measurement_basis_vector.view(2, 2, 2)
                basis_vectors = torch.complex(basis_vectors[..., 0], basis_vectors[..., 1])

                basis_matrices = torch.zeros(2, 2, 2, dtype=torch.complex64, device=rho_k.device)
                basis_matrices[:, 0, 0] = basis_vectors[:, 0].abs() ** 2
                basis_matrices[:, 1, 1] = basis_vectors[:, 1].abs() ** 2
                basis_matrices[:, 0, 1] = basis_vectors[:, 0] * basis_vectors[:, 1].conj()
                basis_matrices[:, 1, 0] = basis_vectors[:, 1] * basis_vectors[:, 0].conj() 
                basis_matrices[0] /= basis_matrices[0].trace()
                basis_matrices[1] /= basis_matrices[1].trace()

                new_measurement = measure(rho_k, (basis_matrices[0], basis_matrices[1]))
                new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(basis.shape[1]*2*2*2)
                new_measurement_predictor_input.append(torch.cat((new_measurement.unsqueeze(0), new_basis_as_vector), dim=-1))

            measurement_predictor_input = torch.stack(new_measurement_predictor_input, dim=0)
            reconstructed_matrices.append(reconstructed_matrix)
            reconstructed_matrix = reconstructed_matrix.view(-1, self.dens_matrix_dim)
        reconstructed_matrices = torch.stack(reconstructed_matrices, dim=1)
        return reconstructed_matrices

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class LSTMMeasurementPredictor(nn.Module):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMMeasurementPredictor, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.basis_sufficient_params = num_qubits*4
        self.dens_matrix_dim = 2 * (4 ** num_qubits)
        # self.measurement_predictor = MLP(layers, 1 + self.basis_dim, hidden_size, self.basis_sufficient_params)
        self.measurement_predictor = nn.LSTMCell(1 + self.basis_dim, hidden_size)
        self.projector = nn.Linear(hidden_size, self.basis_sufficient_params)
        self.matrix_reconstructor = LSTMDensityMatrixReconstructor(1 + self.basis_dim, num_qubits, layers, hidden_size)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
        measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)

        h_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)
        c_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)

        predicted_bases = [basis]
        measurements_with_basis = [measurement_predictor_input]
        for _ in range(self.max_num_measurements - 1):
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            h_i, c_i  = self.measurement_predictor(measurement_predictor_input, (h_i, c_i))
            measurement_basis_vectors = self.projector(h_i)

            new_measurement_predictor_input = []
            new_predicted_bases = []
            for rho_k, measurement_basis_vector in zip(rho, measurement_basis_vectors):
                rho_k = torch.complex(rho_k[0], rho_k[1]).view(*[2, 2]*self.num_qubits)
                basis_vectors = measurement_basis_vector.view(2, 2, 2)
                basis_vectors = torch.complex(basis_vectors[..., 0], basis_vectors[..., 1])

                basis_matrices = torch.zeros(2, 2, 2, dtype=torch.complex64, device=rho_k.device)
                basis_matrices[:, 0, 0] = basis_vectors[:, 0].abs() ** 2
                basis_matrices[:, 1, 1] = basis_vectors[:, 1].abs() ** 2
                basis_matrices[:, 0, 1] = basis_vectors[:, 0] * basis_vectors[:, 1].conj()
                basis_matrices[:, 1, 0] = basis_vectors[:, 1] * basis_vectors[:, 0].conj()
                for j in range(self.num_qubits):
                    basis_matrices[j] /= basis_matrices[j].trace()

                new_measurement = measure(rho_k, (basis_matrices[0], basis_matrices[1]))
                new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(basis.shape[1]*2*2*2)
                new_measurement_predictor_input.append(torch.cat((new_measurement.unsqueeze(0), new_basis_as_vector), dim=-1))
                new_predicted_bases.append(basis_matrices)

            measurement_predictor_input = torch.stack(new_measurement_predictor_input, dim=0)
            measurements_with_basis.append(measurement_predictor_input)
            predicted_bases.append(torch.stack(new_predicted_bases, dim=0))

        measurements_with_basis = torch.stack(measurements_with_basis, dim=1)
        reconstructed_matrices = self.matrix_reconstructor(measurements_with_basis)
        predicted_bases = torch.stack(predicted_bases, dim=1)
        return reconstructed_matrices, predicted_bases
    
    def measurement_inference(self, measurement: torch.Tensor, basis: torch.Tensor, h_i: t.Optional[torch.Tensor] = None, c_i: t.Optional[torch.Tensor] = None):
        '''
        measurement: (batch, 1) 
        basis: (batch, num_qubits, 2, 2)
        '''
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*self.num_qubits*2*2)
        measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)

        if h_i is None:
            h_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)
        if c_i is None:
            c_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)

        h_i, c_i  = self.measurement_predictor(measurement_predictor_input, (h_i, c_i))
        measurement_basis_vectors = self.projector(h_i)

        basis_vectors = measurement_basis_vectors.view(-1, self.num_qubits, 2, 2)

        basis_vectors = torch.complex(basis_vectors[..., 0], basis_vectors[..., 1])

        basis_matrices = torch.zeros(basis_vectors.shape[0], self.num_qubits, 2, 2, dtype=torch.complex64, device=measurement.device)
        basis_matrices[..., 0, 0] = basis_vectors[..., 0].abs() ** 2
        basis_matrices[..., 1, 1] = basis_vectors[..., 1].abs() ** 2
        basis_matrices[..., 0, 1] = basis_vectors[..., 0] * basis_vectors[..., 1].conj()
        basis_matrices[..., 1, 0] = basis_vectors[..., 1] * basis_vectors[..., 0].conj() 
        basis_matrices /= torch.vmap(torch.trace, in_dims=(0, 1))(basis_matrices)
        return basis_matrices
    
    def reconstruction_inference(self, measurements: torch.Tensor, bases: torch.Tensor):
        '''
        measurements: (batch, num_measurements, 1)
        bases: (batch, num_measurements, num_qubits, 2, 2)
        '''
        measurements, bases = measurements
        basis_as_vector = torch.stack((bases.real, bases.imag), dim=-1).view(-1, bases.shape[1], bases.shape[2]*self.num_qubits*2*2)
        measurement_reconstructor_input = torch.cat((measurements, basis_as_vector), dim=-1)
        return self.matrix_reconstructor(measurement_reconstructor_input)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class LSTMMeasurementSelector(nn.Module):
    def __init__(self, num_qubits: int , possible_basis_matrices: t.List[torch.Tensor], layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMMeasurementSelector, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.bases = possible_basis_matrices
        self.dens_matrix_dim = 2 * (4 ** num_qubits)
        # self.measurement_predictor = MLP(layers, 1 + self.basis_dim, hidden_size, self.basis_sufficient_params)
        self.measurement_selector = nn.LSTMCell(1 + self.basis_dim, hidden_size)
        self.projectors = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, len(self.bases)),
            nn.Softmax(dim=-1)
        ) for _ in range(num_qubits)])
        self.matrix_reconstructor = LSTMDensityMatrixReconstructor(1 + self.basis_dim, num_qubits, layers, hidden_size)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*self.num_qubits*2*2)
        measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)

        h_i = torch.randn((measurement.shape[0], self.measurement_selector.hidden_size), device=measurement.device)
        c_i = torch.randn((measurement.shape[0], self.measurement_selector.hidden_size), device=measurement.device)

        measurements_with_basis = [measurement_predictor_input]
        predicted_bases = [basis]
        for i in range(self.max_num_measurements - 1):
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            h_i, c_i  = self.measurement_selector(measurement_predictor_input, (h_i, c_i))
            measurement_basis_probability = torch.stack([projector(h_i) for projector in self.projectors], dim=1) # shape (batch, num_qubits, len(bases))

            new_measurement_predictor_input = []
            new_predicted_bases = []
            for rho_k, measurement_basis_probability_k in zip(rho, measurement_basis_probability):
                rho_k = torch.complex(rho_k[0], rho_k[1]).view(*[2, 2]*self.num_qubits)
                
                # Option 1) argmax
                # basis_matrices = torch.stack([self.bases[i] for i in torch.argmax(measurement_basis_probability_k, dim=-1)], dim=0).to(rho_k.device)
                
                # Option 2) expectation
                # bases_ids = torch.arange(0, len(self.bases), device=rho_k.device).float()
                # most_convenient_bases_ids = torch.matmul(measurement_basis_probability_k, bases_ids[..., None]).round().int()
                # basis_matrices = torch.stack([self.bases[i] for i in most_convenient_bases_ids], dim=0).to(rho_k.device)

                # Option 3) weighted sum of bases
                basis_matrices = torch.stack(self.bases, dim=0).to(rho_k.device) # shape (len(bases), 2, 2)
                measurement_basis_probability_k_expanded = measurement_basis_probability_k.view(self.num_qubits, len(self.bases), 1, 1) # shape (num_qubits, len(bases), 1, 1)
                basis_matrices = torch.sum(basis_matrices * measurement_basis_probability_k_expanded, dim=1) # shape (num_qubits, 2, 2)
                new_predicted_bases.append(basis_matrices)

                new_measurement = measure(rho_k, (basis_matrices[0], basis_matrices[1])) # assuming 2 qubits
                new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(basis.shape[1]*2*2*2)
                new_measurement_predictor_input.append(torch.cat((new_measurement.unsqueeze(0), new_basis_as_vector), dim=-1))

            measurement_predictor_input = torch.stack(new_measurement_predictor_input, dim=0)
            measurements_with_basis.append(measurement_predictor_input)
            predicted_bases.append(torch.stack(new_predicted_bases, dim=0))

        measurements_with_basis = torch.stack(measurements_with_basis, dim=1)
        reconstructed_matrices = self.matrix_reconstructor(measurements_with_basis) # shape (batch, max_num_measurements, 2, 2**num_qubits, 2**num_qubits)
        predicted_bases = torch.stack(predicted_bases, dim=1) # shape (batch, max_num_measurements, num_qubits, 2, 2)
        return reconstructed_matrices, predicted_bases

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
    
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
        return reconstructed_matrices, measurement

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

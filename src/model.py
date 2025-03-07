from functools import reduce
import typing as t
from itertools import product
from math import comb

import numpy as np
import torch
import torch.nn as nn

from src.tomography_utils_torch import measure


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

    def load(self, path: str, **kwargs):
        self.load_state_dict(torch.load(path, **kwargs))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = torch.tensor(x, dtype=torch.float32)
        prediction = self.forward(x)
        prob_vector = torch.cat((1 - prediction, prediction), dim=-1)
        clipped_prob_vector = torch.clamp(prob_vector, 0., 1.)
        return clipped_prob_vector.cpu().detach().numpy()


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
        predicted_bases = [basis]
        for i in range(self.max_num_measurements):
            reconstructed_matrix = self.matrix_reconstructors[i](measurement_with_basis)
            measurement_basis_vectors = self.measurement_predictors[i](measurement_with_basis)
            new_measurement_with_basis = []
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
                basis_matrices[0] /= basis_matrices[0].trace()
                basis_matrices[1] /= basis_matrices[1].trace()

                new_measurement = measure(rho_k, (basis_matrices[0], basis_matrices[1]))
                new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(basis.shape[1]*2*2*2)
                new_measurement_with_basis.append(torch.cat((new_measurement.unsqueeze(0), new_basis_as_vector), dim=-1))
                new_predicted_bases.append(basis_matrices)

            new_measurement_with_basis = torch.stack(new_measurement_with_basis, dim=0)
            measurement_with_basis = torch.cat([measurement_with_basis, new_measurement_with_basis], dim=1)
            reconstructed_matrices.append(reconstructed_matrix)
            predicted_bases.append(torch.stack(new_predicted_bases, dim=0))

        reconstructed_matrices = torch.stack(reconstructed_matrices, dim=1)
        predicted_bases = torch.stack(predicted_bases[:-1], dim=1) # last measurement base is unused
        return reconstructed_matrices, predicted_bases

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class SequentialMeasurementPredictorForConcurrence(SequentialMeasurementPredictor):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(SequentialMeasurementPredictorForConcurrence, self).__init__(num_qubits, layers, hidden_size, max_num_measurements)
        self.matrix_reconstructors = nn.ModuleList([
            ConcurrencePredictor(i *(1 + num_qubits*2*4), num_qubits, layers, hidden_size) for i in range(1, max_num_measurements + 1)
        ])


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
        predicted_bases = [basis]
        for i in range(self.max_num_measurements):
            reconstructor_input = torch.cat((measurement_predictor_input, reconstructed_matrix), dim=-1)
            reconstructed_matrix = self.matrix_reconstructor(reconstructor_input)
            measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
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
                basis_matrices[0] /= basis_matrices[0].trace()
                basis_matrices[1] /= basis_matrices[1].trace()

                new_measurement = measure(rho_k, (basis_matrices[0], basis_matrices[1]))
                new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(basis.shape[1]*2*2*2)
                new_measurement_predictor_input.append(torch.cat((new_measurement.unsqueeze(0), new_basis_as_vector), dim=-1))
                new_predicted_bases.append(basis_matrices)

            measurement_predictor_input = torch.stack(new_measurement_predictor_input, dim=0)
            reconstructed_matrices.append(reconstructed_matrix)
            reconstructed_matrix = reconstructed_matrix.view(-1, self.dens_matrix_dim)
            predicted_bases.append(torch.stack(new_predicted_bases, dim=0))

        predicted_bases = torch.stack(predicted_bases[:-1], dim=1) # last measurement base is unused
        reconstructed_matrices = torch.stack(reconstructed_matrices, dim=1)
        return reconstructed_matrices, predicted_bases

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class LSTMMeasurementProjectorPredictor(nn.Module):
    def __init__(self, num_qubits: int, layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMMeasurementProjectorPredictor, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.projector_sufficient_params = num_qubits*4
        self.dens_matrix_dim = 2 * (4 ** num_qubits)
        self.measurement_projector_predictor = nn.LSTMCell(1 + self.basis_dim, hidden_size)
        self.projector = nn.Linear(hidden_size, self.projector_sufficient_params)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
        measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
        rho_complex = torch.complex(rho[:, 0], rho[:, 1]).view(-1, *[2, 2]*self.num_qubits)

        h_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)
        c_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)

        predicted_bases = [basis]
        measurements_with_basis = [measurement_predictor_input]
        for _ in range(self.max_num_measurements - 1):
            h_i, c_i  = self.measurement_projector_predictor(measurement_predictor_input, (h_i, c_i))
            measurement_vectors = self.projector(h_i)
            basis_matrices = self.construct_projectors(measurement_vectors)
            new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
            new_measurements = torch.stack([measure(rho_complex_k, basis_matrices_k) for rho_complex_k, basis_matrices_k in zip(rho_complex, basis_matrices)]).unsqueeze(-1)
            measurement_predictor_input = torch.cat((new_measurements, new_basis_as_vector), dim=-1)
            measurements_with_basis.append(measurement_predictor_input)
            predicted_bases.append(basis_matrices)

        measurements_with_basis = torch.stack(measurements_with_basis, dim=1)
        reconstructed_matrices = self.matrix_reconstructor(measurements_with_basis)
        predicted_bases = torch.stack(predicted_bases, dim=1)
        return reconstructed_matrices, predicted_bases
    
    def construct_projectors(self, measurement_vectors: torch.Tensor):
        basis_vectors = measurement_vectors.view(-1, self.num_qubits, 2, 2)
        basis_vectors = torch.complex(basis_vectors[..., 0], basis_vectors[..., 1])
        projectors = basis_vectors / torch.linalg.vector_norm(basis_vectors, dim=-1).view(*basis_vectors.shape[:-1], 1)
        return projectors

    def construct_basis_matrices(self, projectors: torch.Tensor):
        # TODO: verify
        basis_vectors = projectors.view(-1, self.num_qubits, 2, 2)
        basis_vectors = torch.complex(basis_vectors[..., 0], basis_vectors[..., 1])
        basis_matrices = torch.zeros(basis_vectors.shape[0], self.num_qubits, 2, 2, dtype=torch.complex64, device=projectors.device)
        basis_matrices[..., 0, 0] = basis_vectors[..., 0].abs() ** 2
        basis_matrices[..., 1, 1] = basis_vectors[..., 1].abs() ** 2
        basis_matrices[..., 0, 1] = basis_vectors[..., 0] * basis_vectors[..., 1].conj()
        basis_matrices[..., 1, 0] = basis_vectors[..., 1] * basis_vectors[..., 0].conj() 
        basis_matrices /= torch.vmap(torch.vmap(torch.trace))(basis_matrices).view(*basis_matrices.shape[:-2], 1, 1)
        return basis_matrices
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, **kwargs):
        self.load_state_dict(torch.load(path, **kwargs))


class LSTMMeasurementPredictor(nn.Module):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16, noise_level: float = 0.1):
        super(LSTMMeasurementPredictor, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.basis_sufficient_params = num_qubits*4
        self.dens_matrix_dim = 2 * (4 ** num_qubits)
        self.noise_level = noise_level
        # self.measurement_predictor = MLP(layers, 1 + self.basis_dim, hidden_size, self.basis_sufficient_params)
        self.measurement_predictor = nn.LSTMCell(1 + self.basis_dim, hidden_size)
        self.projector = nn.Linear(hidden_size, self.basis_sufficient_params)
        self.matrix_reconstructor = LSTMDensityMatrixReconstructor(1 + self.basis_dim, num_qubits, layers, hidden_size)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor, add_noise_to_measurement_basis: bool = False):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
        measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
        rho_complex = torch.complex(rho[:, 0], rho[:, 1]).view(-1, *[2, 2]*self.num_qubits)

        h_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)
        c_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)

        predicted_bases = [basis]
        measurements_with_basis = [measurement_predictor_input]
        for _ in range(self.max_num_measurements - 1):
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            h_i, c_i  = self.measurement_predictor(measurement_predictor_input, (h_i, c_i))
            measurement_basis_vectors = self.projector(h_i)
            if add_noise_to_measurement_basis:
                measurement_basis_vectors += torch.randn_like(measurement_basis_vectors) * self.noise_level
            basis_matrices = self.construct_basis_matrices(measurement_basis_vectors)
            new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
            new_measurements = torch.stack([measure(rho_complex_k, basis_matrices_k) for rho_complex_k, basis_matrices_k in zip(rho_complex, basis_matrices)]).unsqueeze(-1)
            measurement_predictor_input = torch.cat((new_measurements, new_basis_as_vector), dim=-1)
            measurements_with_basis.append(measurement_predictor_input)
            predicted_bases.append(basis_matrices)

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
        basis_matrices = self.construct_basis_matrices(measurement_basis_vectors)
        return basis_matrices, (h_i, c_i)
    
    def reconstruction_inference(self, measurements: torch.Tensor, bases: torch.Tensor):
        '''
        measurements: (batch, num_measurements, 1)
        bases: (batch, num_measurements, num_qubits, 2, 2)
        '''
        measurements, bases = measurements
        basis_as_vector = torch.stack((bases.real, bases.imag), dim=-1).view(-1, bases.shape[1], bases.shape[2]*self.num_qubits*2*2)
        measurement_reconstructor_input = torch.cat((measurements, basis_as_vector), dim=-1)
        return self.matrix_reconstructor(measurement_reconstructor_input)
    
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

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, **kwargs):
        self.load_state_dict(torch.load(path, **kwargs))


class LSTMMeasurementPredictorNoSelectionMeausrements(LSTMMeasurementPredictor):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMMeasurementPredictorNoSelectionMeausrements, self).__init__(num_qubits, layers, hidden_size, max_num_measurements)
        self.measurement_predictor = nn.LSTMCell(self.basis_dim, hidden_size)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor, add_noise_to_measurement_basis: bool = False):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
        reconstructor_input = torch.cat((measurement, basis_as_vector), dim=-1)
        rho_complex = torch.complex(rho[:, 0], rho[:, 1]).view(-1, *[2, 2]*self.num_qubits)

        h_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)
        c_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)

        predicted_bases = [basis]
        measurements_with_basis = [reconstructor_input]
        new_basis_as_vector = basis_as_vector
        for _ in range(self.max_num_measurements - 1):
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            h_i, c_i  = self.measurement_predictor(new_basis_as_vector, (h_i, c_i))
            measurement_basis_vectors = self.projector(h_i)
            if add_noise_to_measurement_basis:
                measurement_basis_vectors += torch.randn_like(measurement_basis_vectors) * self.noise_level
            basis_matrices = self.construct_basis_matrices(measurement_basis_vectors)
            new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
            new_measurements = torch.stack([measure(rho_complex_k, basis_matrices_k) for rho_complex_k, basis_matrices_k in zip(rho_complex, basis_matrices)]).unsqueeze(-1)
            reconstructor_input = torch.cat((new_measurements, new_basis_as_vector), dim=-1)
            measurements_with_basis.append(reconstructor_input)
            predicted_bases.append(basis_matrices)

        measurements_with_basis = torch.stack(measurements_with_basis, dim=1)
        reconstructed_matrices = self.matrix_reconstructor(measurements_with_basis)
        predicted_bases = torch.stack(predicted_bases, dim=1)
        return reconstructed_matrices, predicted_bases
    
    def measurement_inference(self, basis: torch.Tensor, h_i: t.Optional[torch.Tensor] = None, c_i: t.Optional[torch.Tensor] = None):
        '''
        measurement: (batch, 1) 
        basis: (batch, num_qubits, 2, 2)
        '''
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*self.num_qubits*2*2)

        if h_i is None:
            h_i = torch.randn((basis_as_vector.shape[0], self.measurement_predictor.hidden_size), device=basis_as_vector.device)
        if c_i is None:
            c_i = torch.randn((basis_as_vector.shape[0], self.measurement_predictor.hidden_size), device=basis_as_vector.device)

        h_i, c_i  = self.measurement_predictor(basis_as_vector, (h_i, c_i))
        measurement_basis_vectors = self.projector(h_i)
        basis_matrices = self.construct_basis_matrices(measurement_basis_vectors)
        return basis_matrices, (h_i, c_i)


class LSTMAttentionMeasurementPredictor(LSTMMeasurementPredictor):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMAttentionMeasurementPredictor, self).__init__(num_qubits, layers, hidden_size, max_num_measurements)
        self.matrix_reconstructor = LSTMAttentioDensityMatrixnReconstructor(1 + self.basis_dim, num_qubits, layers, hidden_size)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor):
        matrices_w_attention, predicted_bases = super().forward(first_measurement, rho)
        matrices, attention = matrices_w_attention
        reconstructed_rhos = []
        reconstructed_rho = torch.zeros_like(matrices[:, 0])
        for i in range(self.max_num_measurements):
            reconstructed_rho = (1 - attention[:, i]) * reconstructed_rho.detach() + attention[:, i] * matrices[:, i]
            reconstructed_rhos.append(reconstructed_rho)
        return torch.stack(reconstructed_rhos, dim=1), predicted_bases


class LSTMMeasurementPredictorStackedInput(LSTMMeasurementPredictor):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMMeasurementPredictorStackedInput, self).__init__(num_qubits, layers, hidden_size, max_num_measurements)
        self.total_input_dim = (self.basis_dim + 1) * max_num_measurements
        self.measurement_predictor = nn.LSTMCell(self.total_input_dim, hidden_size)
        self.matrix_reconstructor = LSTMDensityMatrixReconstructor(self.total_input_dim, num_qubits, layers, hidden_size)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor, add_noise_to_measurement_basis: bool = False):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
        measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
        rho_complex = torch.complex(rho[:, 0], rho[:, 1]).view(-1, *[2, 2]*self.num_qubits)

        h_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)
        c_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)

        predicted_bases = [basis]
        total_input = torch.cat((measurement_predictor_input, torch.zeros((measurement.shape[0], (self.basis_dim + 1) * (self.max_num_measurements - 1)), device=measurement.device)), dim=-1)
        measurements_with_basis = [total_input]
        for i in range(1, self.max_num_measurements):
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            h_i, c_i  = self.measurement_predictor(total_input, (h_i, c_i))
            measurement_basis_vectors = self.projector(h_i)
            if add_noise_to_measurement_basis:
                measurement_basis_vectors += torch.randn_like(measurement_basis_vectors) * self.noise_level
            basis_matrices = self.construct_basis_matrices(measurement_basis_vectors)
            new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
            new_measurements = torch.stack([measure(rho_complex_k, basis_matrices_k) for rho_complex_k, basis_matrices_k in zip(rho_complex, basis_matrices)]).unsqueeze(-1)
            measurement_predictor_input = torch.cat((new_measurements, new_basis_as_vector), dim=-1)
            total_input = torch.cat(
                (
                    total_input[..., :(self.basis_dim + 1)*i],
                    measurement_predictor_input,
                    torch.zeros((measurement.shape[0], (self.basis_dim + 1) * (self.max_num_measurements - i - 1)), device=measurement.device)
                ),
                dim=-1
            )
            measurements_with_basis.append(total_input)
            predicted_bases.append(basis_matrices)

        measurements_with_basis = torch.stack(measurements_with_basis, dim=1)
        reconstructed_matrices = self.matrix_reconstructor(measurements_with_basis)
        predicted_bases = torch.stack(predicted_bases, dim=1)
        return reconstructed_matrices, predicted_bases


class LSTMMeasurementPredictorBasedOnReconstructedMatrix(nn.Module):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMMeasurementPredictorBasedOnReconstructedMatrix, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.basis_sufficient_params = num_qubits*4
        self.dens_matrix_dim = 2 * (4 ** num_qubits)

        # self.measurement_predictor = MLP(layers, 1 + self.basis_dim, hidden_size, self.basis_sufficient_params)
        self.measurement_predictor = nn.LSTMCell(1 + self.basis_dim + self.dens_matrix_dim, hidden_size)
        self.projector = nn.Linear(hidden_size, self.basis_sufficient_params)
        self.matrix_reconstructor = nn.LSTMCell(1 + self.basis_dim + self.dens_matrix_dim, hidden_size)
        self.matrix_reconstructor_projector = nn.Linear(hidden_size, self.dens_matrix_dim)

    def forward(self, first_measurement: t.Tuple[torch.Tensor, torch.Tensor], rho: torch.Tensor):
        measurement, basis = first_measurement
        basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)
        rho_vector = torch.zeros((measurement.shape[0], self.dens_matrix_dim), device=measurement.device)
        first_reconstructor_input = torch.cat((measurement, basis_as_vector, rho_vector), dim=-1)

        matrix_reconstructor_h_i = torch.randn((measurement.shape[0], self.matrix_reconstructor.hidden_size), device=measurement.device)
        matrix_reconstructor_c_i = torch.randn((measurement.shape[0], self.matrix_reconstructor.hidden_size), device=measurement.device)
        matrix_reconstructor_h_i, matrix_reconstructor_c_i = self.matrix_reconstructor(first_reconstructor_input, (matrix_reconstructor_h_i, matrix_reconstructor_c_i))
        reconstructed_rho = self.matrix_reconstructor_projector(matrix_reconstructor_h_i).view(first_reconstructor_input.shape[0], 2, 2 ** self.num_qubits, 2 ** self.num_qubits)
        lstm_input = torch.cat((measurement, basis_as_vector, reconstructed_rho.view(-1, self.dens_matrix_dim).detach()), dim=-1)

        measurement_predictor_h_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)
        measurement_predictor_c_i = torch.randn((measurement.shape[0], self.measurement_predictor.hidden_size), device=measurement.device)

        rho_complex = torch.complex(rho[:, 0], rho[:, 1]).view(-1, *[2, 2]*self.num_qubits)

        predicted_bases = [basis]
        reconstructed_matrices = [reconstructed_rho]
        for _ in range(self.max_num_measurements - 1):
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            measurement_predictor_h_i, measurement_predictor_c_i  = self.measurement_predictor(lstm_input, (measurement_predictor_h_i, measurement_predictor_c_i))
            measurement_basis_vectors = self.projector(measurement_predictor_h_i)
            basis_matrices = self.construct_basis_matrices(measurement_basis_vectors)
            new_basis_as_vector = torch.stack((basis_matrices.real, basis_matrices.imag), dim=-1).view(-1, basis.shape[1]*2*2*2)

            new_measurements = torch.stack([measure(rho_complex_k, basis_matrices_k) for rho_complex_k, basis_matrices_k in zip(rho_complex, basis_matrices)]).unsqueeze(-1)
            lstm_input = torch.cat((new_measurements, new_basis_as_vector, reconstructed_rho.view(-1, self.dens_matrix_dim).detach()), dim=-1)
            matrix_reconstructor_h_i, matrix_reconstructor_c_i = self.matrix_reconstructor(lstm_input, (matrix_reconstructor_h_i, matrix_reconstructor_c_i))
            reconstructed_rho = self.matrix_reconstructor_projector(matrix_reconstructor_h_i).view(lstm_input.shape[0], 2, 2 ** self.num_qubits, 2 ** self.num_qubits)
            
            reconstructed_matrices.append(reconstructed_rho)
            predicted_bases.append(basis_matrices)

        reconstructed_matrices = torch.stack(reconstructed_matrices, dim=1)
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

        basis_matrices = self.construct_basis_matrices(measurement_basis_vectors)
        return basis_matrices, (h_i, c_i)
    
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
    

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class LSTMMeasurementPredictorForConcurrence(LSTMMeasurementPredictor):
    def __init__(self, num_qubits: int , layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMMeasurementPredictorForConcurrence, self).__init__(num_qubits, layers, hidden_size, max_num_measurements)
        self.matrix_reconstructor = LSTMConcurrencePredictor(1 + self.basis_dim, num_qubits, layers, hidden_size, bias=True)
        


class LSTMMeasurementSelector(nn.Module):
    def __init__(self, num_qubits: int , possible_basis_matrices: t.List[torch.Tensor], layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMMeasurementSelector, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.bases = possible_basis_matrices
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

                new_measurement = measure(rho_k, basis_matrices)
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


class LSTMDiscreteMeasurementSelector(nn.Module):
    def __init__(self, num_qubits: int , possible_basis_matrices: t.List[torch.Tensor], layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMDiscreteMeasurementSelector, self).__init__()
        self.max_num_measurements = max_num_measurements
        self.num_qubits = num_qubits
        self.basis_dim = 2 * (num_qubits*4)
        self.bases = torch.stack(possible_basis_matrices, dim=0)
        self.multiqubit_bases_ids = list(product(range(len(self.bases)), repeat=self.num_qubits))
        self.measurement_selector = nn.LSTMCell(1 + self.basis_dim, hidden_size)
        self.projectors = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, len(self.bases)),
            nn.Softmax(dim=-1)
        ) for _ in range(num_qubits)])
        self.matrix_reconstructor = LSTMDensityMatrixReconstructor(1 + self.basis_dim, num_qubits, layers, hidden_size, bias=True)

    def forward(self, all_measurements: t.List[t.Tuple[torch.Tensor, torch.Tensor]], rho: torch.Tensor):
        # in current implementation it seems that selector starts from the last measurment of the Kwiat basis, while reconstructor starts from the first one -- corrected in fixed model
        target_measurements_with_basis = []
        for measurment_tuple in all_measurements:
            measurement, basis = measurment_tuple
            basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, self.num_qubits*2*2*2)
            measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
            target_measurements_with_basis.append(measurement_predictor_input)

        first_measurement = all_measurements[0]
        measurement, basis = first_measurement
        h_i = torch.randn((measurement.shape[0], self.measurement_selector.hidden_size), device=measurement.device)
        c_i = torch.randn((measurement.shape[0], self.measurement_selector.hidden_size), device=measurement.device)

        predicted_measurements_with_basis = [target_measurements_with_basis[0]]
        first_measurements_basis_probability = torch.zeros(measurement.shape[0], self.num_qubits, len(self.bases), device=measurement.device)
        first_measurements_basis_probability[:, :, 0] = 1.0
        predicted_bases_probabilites = [first_measurements_basis_probability]
        basis_matrices = self.bases.to(rho.device) # shape (len(bases), 2, 2)
        
        target_measurements_matrices = torch.stack([
            self.matrix_reconstructor(target_measurement_with_basis.unsqueeze(1))[:, -1]
            for target_measurement_with_basis in target_measurements_with_basis
        ], dim=1)
        total_target_measurements_matrices = [target_measurements_matrices]
        measurement_predictor_input = target_measurements_with_basis[0] # it was unused before the fix!!

        used_measurement_bases_ids: t.List[t.Set[int]] = [{0} for _ in range(measurement.shape[0])]
        
        for i in range(self.max_num_measurements - 1):
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            h_i, c_i  = self.measurement_selector(measurement_predictor_input, (h_i, c_i))
            measurement_basis_probability = torch.stack([projector(h_i) for projector in self.projectors], dim=1) # shape (batch, num_qubits, len(bases))

            new_measurement_predictor_input = []
            for k, (rho_k, measurement_basis_probability_k) in enumerate(zip(rho, measurement_basis_probability)):
                rho_k = torch.complex(rho_k[0], rho_k[1]).view(*[2, 2]*self.num_qubits)
                
                # Option 1) argmax
                # selected_basis_indices = torch.argmax(measurement_basis_probability_k, dim=-1) # shape (num_qubits)
                
                # Option 2) Filtered argmax (choosing highest probability from those not chosen yet)
                probabilites = reduce(torch.kron, torch.unbind(measurement_basis_probability_k, dim=0)) # shape (len(bases) ** num_qubits)
                sorted_indices = torch.argsort(probabilites, descending=True, dim=-1)
                # get the highest probability index that was not used yet
                selected_multiqubit_basis_id = self._get_first_index_not_in_set(sorted_indices, used_measurement_bases_ids[k])
                used_measurement_bases_ids[k].add(selected_multiqubit_basis_id)
                selected_basis_indices = torch.tensor(self.multiqubit_bases_ids[selected_multiqubit_basis_id]).to(measurement.device)

                discrete_basis_matrices = basis_matrices[selected_basis_indices]                

                new_measurement = measure(rho_k, discrete_basis_matrices) # old: (discrete_basis_matrices[0], discrete_basis_matrices[1])) # assuming 2 qubits
                new_basis_as_vector = torch.stack((discrete_basis_matrices.real, discrete_basis_matrices.imag), dim=-1).view(self.num_qubits*2*2*2)
                new_measurement_predictor_input.append(torch.cat((new_measurement.unsqueeze(0), new_basis_as_vector), dim=-1))

            measurements_with_basis = torch.stack(predicted_measurements_with_basis, dim=1)
            target_measurements_matrices = []
            for target_measurement_with_basis in target_measurements_with_basis:
                current_measurements_with_basis = torch.cat((measurements_with_basis, target_measurement_with_basis.unsqueeze(1)), dim=1)
                target_measurement_matrix = self.matrix_reconstructor(current_measurements_with_basis)[:, -1] # shape (batch, max_num_measurements, 2, 2**num_qubits, 2**num_qubits)
                target_measurements_matrices.append(target_measurement_matrix)
            target_measurements_matrices = torch.stack(target_measurements_matrices, dim=1) # shape (batch, len(all_measurements), 2, 2**num_qubits, 2**num_qubits)

            measurement_predictor_input = torch.stack(new_measurement_predictor_input, dim=0)
            predicted_measurements_with_basis.append(measurement_predictor_input)
            predicted_bases_probabilites.append(measurement_basis_probability)
            total_target_measurements_matrices.append(target_measurements_matrices)

        predicted_bases_probabilites = torch.stack(predicted_bases_probabilites, dim=1) # shape (batch, max_num_measurements, num_qubits, len(bases))
        predicted_measurements_with_basis = torch.stack(predicted_measurements_with_basis, dim=1)
        predicted_measurements_matrices = self.matrix_reconstructor(predicted_measurements_with_basis) #.detach()) # shape (batch, max_num_measurements, 2, 2**num_qubits, 2**num_qubits)
        total_target_measurements_matrices = torch.stack(total_target_measurements_matrices, dim=1) # shape (batch, max_num_measurements, len(all_measurements), 2, 2**num_qubits, 2**num_qubits)
        return predicted_measurements_matrices, predicted_bases_probabilites, total_target_measurements_matrices


    def _get_first_index_not_in_set(self, ids: t.List[torch.Tensor], set: t.Set[int]) -> int:
        for i in ids:
            if i.item() not in set:
                return i.item()
        return i.item()


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

        h_i, c_i  = self.measurement_selector(measurement_predictor_input, (h_i, c_i))
        measurement_basis_probability = torch.stack([projector(h_i) for projector in self.projectors], dim=1) # shape (batch, num_qubits, len(bases))
        return measurement_basis_probability, (h_i, c_i)

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


class LSTMDiscreteMeasurementSelectorOptimized(LSTMDiscreteMeasurementSelector):
    def forward(self, all_measurements: t.List[t.Tuple[torch.Tensor, torch.Tensor]], rho: torch.Tensor, predict_target_basis: bool = False, add_noise_while_selecting_basis: bool = False):
        target_measurements_with_basis = []
        for measurment_tuple in all_measurements:
            measurement, basis = measurment_tuple
            basis_as_vector = torch.stack((basis.real, basis.imag), dim=-1).view(-1, self.num_qubits*2*2*2)
            measurement_predictor_input = torch.cat((measurement, basis_as_vector), dim=-1)
            target_measurements_with_basis.append(measurement_predictor_input)

        first_measurement = all_measurements[0]
        measurement, basis = first_measurement
        h_i = torch.randn((measurement.shape[0], self.measurement_selector.hidden_size), device=measurement.device)
        c_i = torch.randn((measurement.shape[0], self.measurement_selector.hidden_size), device=measurement.device)

        predicted_measurements_with_basis = [target_measurements_with_basis[0]]
        first_measurements_basis_probability = torch.zeros(measurement.shape[0], self.num_qubits, len(self.bases), device=measurement.device)
        first_measurements_basis_probability[:, :, 0] = 1.0
        predicted_bases_probabilities = [first_measurements_basis_probability]
        basis_matrices = self.bases.to(rho.device) # shape (len(bases), 2, 2)
        
        total_target_basis_ids = []
        measurement_predictor_input = target_measurements_with_basis[0] # it was unused before the fix!!

        used_measurement_bases_ids: t.List[t.Set[int]] = [{0} for _ in range(measurement.shape[0])]
        rho_complex = torch.complex(rho[:, 0], rho[:, 1]).view(-1, *[2, 2]*self.num_qubits)

        
        for i in range(self.max_num_measurements - 1):
            # measurement_basis_vectors = self.measurement_predictor(measurement_predictor_input)
            h_i, c_i  = self.measurement_selector(measurement_predictor_input, (h_i, c_i))
            measurement_basis_probability = torch.stack([projector(h_i) for projector in self.projectors], dim=1) # shape (batch, num_qubits, len(bases))
            probabilites = torch.stack(
                [reduce(torch.kron, torch.unbind(measurement_basis_probability_k, dim=0)) for measurement_basis_probability_k in measurement_basis_probability],
                dim=0 
            )
            sorted_indices = torch.argsort(probabilites, descending=True, dim=-1)
            measurement_predictor_input = torch.stack([
                self._get_new_measurement_predictor_input(sorted_indices_k, used_measurement_bases_ids_k, rho_k, basis_matrices)
                for sorted_indices_k, used_measurement_bases_ids_k, rho_k in zip(sorted_indices, used_measurement_bases_ids, rho_complex)
            ], dim=0)

            measurements_with_basis = torch.stack(predicted_measurements_with_basis, dim=1)
            if predict_target_basis:
                target_reconstruction_losses = []
                with torch.no_grad():
                    for target_measurement_with_basis in target_measurements_with_basis:
                        current_measurements_with_basis = torch.cat((measurements_with_basis, target_measurement_with_basis.unsqueeze(1)), dim=1)
                        target_measurement_matrix = self.matrix_reconstructor(current_measurements_with_basis)[:, -1] # shape (batch, max_num_measurements, 2, 2**num_qubits, 2**num_qubits)
                        reconstruction_losses = torch.nn.functional.mse_loss(
                            target_measurement_matrix,  # skipping the first basis as it is always chosen as the first measurement
                            rho,
                            reduction='none'
                        ).mean(dim=(-1, -2, -3))
                        if add_noise_while_selecting_basis:
                            reconstruction_losses = reconstruction_losses + torch.randn_like(reconstruction_losses) * 1e-4
                        target_reconstruction_losses.append(reconstruction_losses)
                    target_reconstruction_losses = torch.stack(target_reconstruction_losses, dim=1) # shape (batch, len(all_measurements))
                target_basis_ids = torch.argmin(target_reconstruction_losses, dim=1).detach() # shape (batch,)
            else:
                target_basis_ids = torch.zeros(measurement.shape[0], device=measurement.device, dtype=torch.int64)

            predicted_measurements_with_basis.append(measurement_predictor_input)
            predicted_bases_probabilities.append(measurement_basis_probability)
            total_target_basis_ids.append(target_basis_ids)

        predicted_bases_probabilities = torch.stack(predicted_bases_probabilities, dim=1) # shape (batch, max_num_measurements, num_qubits, len(bases))
        predicted_measurements_with_basis = torch.stack(predicted_measurements_with_basis, dim=1)
        
        predicted_measurements_matrices = self.matrix_reconstructor(predicted_measurements_with_basis) #.detach()) # shape (batch, max_num_measurements, 2, 2**num_qubits, 2**num_qubits)
        total_target_basis_ids = torch.stack(total_target_basis_ids, dim=1) # shape (batch, max_num_measurements, 1)
        return predicted_measurements_matrices, predicted_bases_probabilities, total_target_basis_ids
    

    def _get_new_measurement_predictor_input(self, sorted_indices: torch.Tensor, used_measurement_bases_ids: t.Set[int], rho: torch.Tensor, basis_matrices: torch.Tensor):
        selected_multiqubit_basis_id = self._get_first_index_not_in_set(sorted_indices, used_measurement_bases_ids)
        used_measurement_bases_ids.add(selected_multiqubit_basis_id)
        selected_basis_indices = torch.tensor(self.multiqubit_bases_ids[selected_multiqubit_basis_id]).to(rho.device)
        discrete_basis_matrices = basis_matrices[selected_basis_indices]   
        new_measurement = measure(rho, discrete_basis_matrices) 
        new_basis_as_vector = torch.stack((discrete_basis_matrices.real, discrete_basis_matrices.imag), dim=-1).view(self.num_qubits*2*2*2)
        new_measurement_predictor_input = torch.cat((new_measurement.unsqueeze(0), new_basis_as_vector), dim=-1)
        return new_measurement_predictor_input
    

    def _get_first_index_not_in_set(self, ids: t.List[torch.Tensor], set: t.Set[int]) -> int:
        for i in ids:
            if i.item() not in set:
                return i.item()
        return i.item()


class LSTMDiscreteMeasurementSelectorForConcurrence(LSTMDiscreteMeasurementSelector):
    def __init__(self, num_qubits: int , possible_basis_matrices: t.List[torch.Tensor], layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(LSTMDiscreteMeasurementSelectorForConcurrence, self).__init__(num_qubits, possible_basis_matrices, layers, hidden_size, max_num_measurements)
        self.matrix_reconstructor = LSTMConcurrencePredictor(1 + self.basis_dim, num_qubits, layers, hidden_size, bias=True)


class TomographyCorrectionsLSTMDiscreteMeasurementSelector(LSTMDiscreteMeasurementSelector):
    def __init__(self, num_qubits: int , possible_basis_matrices: t.List[torch.Tensor], layers: int = 2, hidden_size: int = 16, max_num_measurements: int = 16):
        super(TomographyCorrectionsLSTMDiscreteMeasurementSelector, self).__init__(num_qubits, possible_basis_matrices, layers, hidden_size, max_num_measurements)
        self.num_gammas = 4 ** num_qubits
        self.matrix_reconstructor = LSTMTomographyCorrectionsPredictor(1 + self.basis_dim, self.num_gammas,  layers, hidden_size)


class DensityMatrixReconstructor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16):
        super(DensityMatrixReconstructor, self).__init__()
        self.num_qubits = num_qubits
        self.mlp = MLP(layers, input_dim, hidden_size, 2 * (4 ** num_qubits))

    def forward(self, measurement_with_basis: torch.Tensor):
        return self.mlp(measurement_with_basis).view(-1, 2, 2 ** self.num_qubits, 2 ** self.num_qubits)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class GammasReconstructor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16, num_gammas: int = 1):
        super(GammasReconstructor, self).__init__()
        self.num_qubits = num_qubits
        self.num_gammas = num_gammas
        self.mlp = MLP(layers, input_dim, hidden_size, num_gammas * 2 * (4 ** num_qubits))

    def forward(self, measurement_with_basis: torch.Tensor):
        return self.mlp(measurement_with_basis).view(-1, self.num_gammas, 2, 2 ** self.num_qubits, 2 ** self.num_qubits)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class TomographyCorrectionsPredictor(nn.Module):
    def __init__(self, input_dim: int, num_measurements: int,  num_gammas: int = 1, layers: int = 2, hidden_size: int = 16):
        super(TomographyCorrectionsPredictor, self).__init__()
        self.num_measurements = num_measurements
        self.num_gammas = num_gammas
        self.mlp = MLP(layers, input_dim, hidden_size, 2 * num_gammas * (num_measurements + 1))

    def forward(self, measurement_with_basis: torch.Tensor): # maybe gamma should be also added as input?
        corrections = self.mlp(measurement_with_basis)
        inverse_corrections = corrections[..., :2*self.num_measurements*self.num_gammas]
        inverse_corrections = inverse_corrections.view(*inverse_corrections.shape[:-1], 2, self.num_measurements, self.num_gammas)
        r_corrections = corrections[..., 2*self.num_measurements*self.num_gammas:]
        r_corrections = r_corrections.view(*r_corrections.shape[:-1], 2, self.num_gammas)
        return inverse_corrections, r_corrections
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class TomographyM2CorrectionsPredictor(nn.Module):
    def __init__(self, input_dim: int, num_measurements: int,  num_gammas: int = 1, layers: int = 2, hidden_size: int = 16):
        super(TomographyM2CorrectionsPredictor, self).__init__()
        self.num_measurements = num_measurements
        self.num_gammas = num_gammas
        self.num_m2_corrections = 2*num_gammas*(comb(num_measurements, 2) + num_measurements)
        self.mlp = MLP(layers, input_dim, hidden_size, self.num_m2_corrections + 2 * num_gammas * (num_measurements + 1))

    def forward(self, measurement_with_basis: torch.Tensor): # maybe gamma should be also added as input?
        corrections = self.mlp(measurement_with_basis)
        m2_corrections = corrections[..., :self.num_m2_corrections].view(*corrections.shape[:-1], 2, -1, self.num_gammas)
        linear_corections = corrections[..., self.num_m2_corrections:]
        inverse_corrections = linear_corections[..., :2*self.num_measurements*self.num_gammas]
        inverse_corrections = inverse_corrections.view(*inverse_corrections.shape[:-1], 2, self.num_measurements, self.num_gammas)
        r_corrections = linear_corections[..., 2*self.num_measurements*self.num_gammas:]
        r_corrections = r_corrections.view(*r_corrections.shape[:-1], 2, self.num_gammas)
        return m2_corrections, inverse_corrections, r_corrections
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class LSTMTomographyCorrectionsPredictor(nn.Module):
    def __init__(self, input_dim: int, num_gammas: int = 1, layers: int = 2, hidden_size: int = 16):
        super(LSTMTomographyCorrectionsPredictor, self).__init__()
        self.num_gammas = num_gammas
        self.lstm = nn.LSTM(input_dim, hidden_size, layers, batch_first=True)
        self.projector = nn.Linear(hidden_size, 2 * num_gammas * (1 + 1))

    def forward(self, measurement_with_basis: torch.Tensor): # maybe gamma should be also added as input?
        hidden_info, _ = self.lstm(measurement_with_basis)
        corrections = self.projector(hidden_info)
        inverse_corrections = corrections[..., :2*self.num_gammas]
        inverse_corrections = inverse_corrections.view(*inverse_corrections.shape[:-1], 2, 1, self.num_gammas)
        r_corrections = corrections[..., 2*self.num_gammas:]
        r_corrections = r_corrections.view(*r_corrections.shape[:-1], 2, 1, self.num_gammas)
        return torch.cat((inverse_corrections, r_corrections), dim=-2)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class ConcurrencePredictor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16):
        super(ConcurrencePredictor, self).__init__()
        self.num_qubits = num_qubits
        self.mlp = MLP(layers, input_dim, hidden_size, 1)

    def forward(self, measurement_with_basis: torch.Tensor):
        return self.mlp(measurement_with_basis).view(-1, 1)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    

class LSTMDensityMatrixReconstructor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16, bias: bool = True):
        super(LSTMDensityMatrixReconstructor, self).__init__()
        self.num_qubits = num_qubits
        self.lstm = nn.LSTM(input_dim, hidden_size, layers, proj_size=2 * (4 ** num_qubits), batch_first=True, bias=bias)

    def reinitialize(self):
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)

    def forward(self, measurements_with_basis: torch.Tensor):
        out, _ = self.lstm(measurements_with_basis)
        return out.view(measurements_with_basis.shape[0], measurements_with_basis.shape[1], 2, 2 ** self.num_qubits, 2 ** self.num_qubits)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))


class LSTMAttentioDensityMatrixnReconstructor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16, bias: bool = True):
        super(LSTMAttentioDensityMatrixnReconstructor, self).__init__()
        self.num_qubits = num_qubits
        self.lstm = nn.LSTM(input_dim, hidden_size, layers, proj_size=2* 2 * (4 ** num_qubits), batch_first=True, bias=bias)

    def reinitialize(self):
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)

    def forward(self, measurements_with_basis: torch.Tensor):
        out, _ = self.lstm(measurements_with_basis)
        value_w_attention = out.view(measurements_with_basis.shape[0], measurements_with_basis.shape[1], 2, 2, 2 ** self.num_qubits, 2 ** self.num_qubits)
        value = value_w_attention[:, :, 0]
        attention = torch.sigmoid(value_w_attention[:, :, 1])
        return value, attention

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))



class LSTMConcurrencePredictor(nn.Module):
    def __init__(self, input_dim: int, num_qubits: int, layers: int = 2, hidden_size: int = 16, bias: bool = True):
        super(LSTMConcurrencePredictor, self).__init__()
        self.num_qubits = num_qubits
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_size, layers, proj_size=1, batch_first=True, bias=bias)

    def reinitialize(self):
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)

    def forward(self, measurements_with_basis: torch.Tensor):
        # batch_normed_measurements_with_basis = torch.stack([self.batch_norm(measurements_with_basis[:, i]) for i in range(measurements_with_basis.shape[1])], dim=1)
        out, _ = self.lstm(measurements_with_basis)
        return out.view(measurements_with_basis.shape[0], measurements_with_basis.shape[1], 1)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
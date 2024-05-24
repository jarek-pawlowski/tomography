import typing as t
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils_measure import Measurement, Kwiat, Pauli, Pauli_c


DICTIONARY_NAME = 'dictionary.txt'
MATRICES_DIR_NAME = 'matrices'


class DensityMatrixDataset(Dataset):
    def __init__(self, root_path: str) -> None:
        self.dict = self.load_dict(root_path + DICTIONARY_NAME)   
        self.root_dir = root_path 

    def __len__(self) -> int:
        return len(self.dict)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        filename = self.read_filename(idx)
        matrix = self.read_matrix(filename)
        tensor = self.convert_numpy_matrix_to_tensor(matrix)
        label = float(self.dict[idx][1])
        label = torch.tensor(label).unsqueeze(-1)
        return (tensor, label)

    def convert_numpy_matrix_to_tensor(self, matrix: np.ndarray) -> torch.Tensor:
        matrix_r = np.real(matrix)
        matrix_im = np.imag(matrix)
        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0)).float()
        return tensor

    def read_matrix(self, filename):
        matrix_name = os.path.join(self.root_dir, MATRICES_DIR_NAME, filename)
        matrix = np.load(matrix_name)
        return matrix

    def read_filename(self, idx):
        filename = f"{self.dict[idx][0]}.npy"
        if not filename.startswith('dens'):
            filename = 'dens' + filename
        return filename

    def load_dict(self, filepath: str) -> t.List[t.List[str]]:
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()
        parsed_data = [row.rstrip("\n").split(', ') for row in data]
        return parsed_data
    
class VectorDataset(Dataset):
    def __init__(self, root_path: str, num_qubits : int) -> None:
        self.root_dir = root_path
        self.num_qubits = num_qubits
        self.dict = self.load_dict(root_path + DICTIONARY_NAME)   

    def __len__(self) -> int:
        return len(self.dict)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        filename = self.read_filename(idx)
        matrix = self.read_vector(filename)
        tensor = self.convert_numpy_vector_to_tensor(matrix)
        label = float(idx)
        label = torch.tensor(label).unsqueeze(-1)
        return (tensor, label)

    def convert_numpy_vector_to_tensor(self, vector: np.ndarray) -> torch.Tensor:
        vector_r = np.real(vector)
        vector_im = np.imag(vector)
        tensor = torch.from_numpy(np.stack((vector_r, vector_im), axis=0)).float()
        return tensor

    def read_vector(self, filename):
        vector_name = os.path.join(self.root_dir, filename)
        vector = np.load(vector_name)
        return vector

    def read_filename(self, idx):
        filename = f"train/{self.num_qubits}_tensor_state_{idx}.npy"
        #if not filename.startswith(str(self.num_qubits)):
            #raise ValueError("Wrong filename: check the dataset!")
            #print(filename)
        return filename
    
    def load_dict(self, filepath: str) -> t.List[t.List[str]]:
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()
        parsed_data = [row.rstrip("\n").split(', ') for row in data]
        return parsed_data

class VectorDensityMatrixDataset(DensityMatrixDataset):
    def __init__(self, root_path: str) -> None:
        super().__init__(root_path)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        rho, label = super().__getitem__(idx)
        upper_real = torch.triu(rho[0], diagonal=0)
        lower_imag = torch.tril(rho[1], diagonal=-1)
        matrix = upper_real + lower_imag
        return (torch.reshape(matrix, (-1,)), label)


class MeasurementDataset(DensityMatrixDataset):
    def __init__(self, num_qubits: int, root_path: str, return_density_matrix: bool = False, data_limit: t.Optional[int] = None, binary_label: bool = False, mask_measurements: t.Optional[t.List] = None) -> None:
        super().__init__(root_path)
        self.num_qubits = num_qubits
        self.measurement = Measurement(Kwiat, 2)
        self.return_density_matrix = return_density_matrix
        self.binary_label = binary_label
        self.mask_measurements = mask_measurements
        if data_limit is not None:
            self.dict = self.dict[:data_limit]

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        filename = self.read_filename(idx)
        matrix = self.read_matrix(filename)
        rho = self.convert_numpy_matrix_to_tensor(matrix)

        # reshape density matrix from (4, 4) to (2, 2, 2, 2)
        matrix = matrix.reshape(tuple(2 for _ in range(2*self.num_qubits)))
        measurements = self._get_all_measurements(matrix)
        if self.mask_measurements is not None:
            measurements = np.array([measurement if i not in self.mask_measurements else np.random.rand() for i, measurement in enumerate(measurements)])
        tensor = torch.from_numpy(measurements).float()
        label = float(self.dict[idx][1])
        label = torch.tensor(label).unsqueeze(-1)
        if self.binary_label:
            label = torch.where(label > 1.e-6, torch.tensor(1.), torch.tensor(0.))

        if not self.return_density_matrix:
            return (tensor, label)
        return (rho, tensor, label)
    
    def _get_all_measurements(self, rho_in: np.ndarray) -> np.ndarray:
        m_all = np.array([[self.measurement.measure(rho_in, [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
        return m_all

class MeasurementVectorDataset(VectorDataset):
    def __init__(self, num_qubits: int, root_path: str, return_density_matrix: bool = False, data_limit: t.Optional[int] = None, binary_label: bool = False, mask_measurements: t.Optional[t.List] = None) -> None:
        super().__init__(root_path, num_qubits)
        self.num_qubits = num_qubits
        self.measurement = Measurement(Pauli, num_qubits, basis_c=Pauli_c)
        self.return_density_matrix = return_density_matrix
        self.binary_label = binary_label
        self.mask_measurements = mask_measurements

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        filename = self.read_filename(idx)
        vector = self.read_vector(filename)
        psi = self.convert_numpy_vector_to_tensor(vector)

        # reshape 2 qubits vector matrix from (1, 4) to (2, 2)
        vector = vector.reshape(tuple(2 for _ in range(self.num_qubits)))
        measurements = self._get_all_measurements(vector)
        if self.mask_measurements is not None:
            measurements = np.array([measurement if i not in self.mask_measurements else np.random.rand() for i, measurement in enumerate(measurements)])
        tensor = torch.from_numpy(measurements).float()
        label = float(idx)
        label = torch.tensor(label).unsqueeze(-1)
        if self.binary_label:
            label = torch.where(label > 1.e-6, torch.tensor(1.), torch.tensor(0.))

        if not self.return_density_matrix:
            return (tensor, label)
        return (psi, tensor, label)
    
    def _get_all_measurements(self, psi_in: np.ndarray) -> np.ndarray:
        m_all = np.array([[self.measurement.measure_single_pure(psi_in, i, j) for j in [0,1,2]] for i in range(self.num_qubits)]).flatten()
        return m_all
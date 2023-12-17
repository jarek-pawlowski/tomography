import typing as t
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils_measure import Measurement, Kwiat


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
    

class MeasurementDataset(DensityMatrixDataset):
    def __init__(self, root_path: str, return_density_matrix: bool = False, data_limit: t.Optional[int] = None) -> None:
        super().__init__(root_path)
        self.measurement = Measurement(Kwiat, 2)
        self.return_density_matrix = return_density_matrix
        if data_limit is not None:
            self.dict = self.dict[:data_limit]

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        filename = self.read_filename(idx)
        matrix = self.read_matrix(filename)
        rho = self.convert_numpy_matrix_to_tensor(matrix)

        # reshape density matrix from (4, 4) to (2, 2, 2, 2)
        matrix = matrix.reshape((2, 2, 2, 2))
        measurements = self._get_all_measurements(matrix)
        tensor = torch.from_numpy(measurements).float()
        label = float(self.dict[idx][1])
        label = torch.tensor(label).unsqueeze(-1)

        if not self.return_density_matrix:
            return (tensor, label)
        return (rho, tensor, label)
    
    def _get_all_measurements(self, rho_in: np.ndarray) -> np.ndarray:
        m_all = np.array([[self.measurement.measure(rho_in, [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
        return m_all

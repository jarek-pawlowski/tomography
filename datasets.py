import typing as t
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils_measure import Measurement, Kwiat


DICTIONARY_NAME = 'dictionary.txt'
MATRICES_DIR_NAME = 'matrices'


class DensityMatrixDataset(Dataset):
    def __init__(self, root_path: str) -> None:
        self.dict = self.load_dict(root_path + DICTIONARY_NAME)    

    def __len__(self) -> int:
        return len(self.dict)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        filename = self.read_filename(idx)
        matrix = self.read_matrix(filename)
        tensor = self.convert_numpy_matrix_to_tensor(matrix)
        label = self.dict[idx][1]
        label = torch.tensor(label)
        return (tensor, label)

    def convert_numpy_matrix_to_tensor(self, matrix: np.ndarray) -> torch.Tensor:
        matrix_r = np.real(matrix)
        matrix_im = np.imag(matrix)
        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0))
        return tensor

    def read_matrix(self, filename):
        matrix_name = os.path.join(self.root_dir, MATRICES_DIR_NAME, filename)
        matrix = np.load(matrix_name)
        return matrix

    def read_filename(self, idx):
        filename = f"{self.dictionary[idx][0]}.npy"
        if not filename.startswith('dens'):
            filename = 'dens' + filename
        return filename

    def load_dict(self, filepath: str) -> t.List[t.List[str]]:
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()
        parsed_data = [row.rstrip("\n").split(', ') for row in data]
        return parsed_data
    

class MeasurementDataset(DensityMatrixDataset):
    def __init__(self, root_path: str) -> None:
        super().__init__(root_path)
        self.measurement = Measurement(Kwiat, 2)

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        filename = self.read_filename(idx)
        matrix = self.read_matrix(filename)
        tensor = self.convert_numpy_matrix_to_tensor(matrix)

        # reshape density matrix from (4, 4) to (2, 2, 2, 2)
        matrix = matrix.reshape((2, 2, 2, 2))
        measurements = self._get_all_measurements(matrix)
        label = torch.from_numpy(measurements)
        return (tensor, label)
    
    def _get_all_measurements(self, rho_in: np.ndarray) -> np.ndarray:
        m_all = np.array([[self.measurement.measure(rho_in, [i,j]) for j in [0,1,2,3]] for i in [0,1,2,3]]).flatten()
        return m_all

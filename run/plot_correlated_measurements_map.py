import sys
sys.path.append('./')

import torch
from torch.utils.data import DataLoader

from src.utils_measure import Kwiat
from src.datasets import MeasurementDataset
from src.model import LSTMMeasurementPredictor, LSTMMeasurementSelector
from src.torch_utils import reconstruct_rho
from src.logging import plot_matrices

batch_size = 128
test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat.basis]

# create model
model_name = 'full_lstm_measure_basis'
model_save_path = f'./models/{model_name}.pt'

# model_params = {
#     'num_qubits': 2,
#     'possible_basis_matrices': basis_matrices, # 'Kwiat' basis matrices
#     'layers': 6,
#     'hidden_size': 128,
#     'max_num_measurements': 16
# }
# model = LSTMMeasurementSelector(**model_params)

model_params = {
    'num_qubits': 2,
    'layers': 6,
    'hidden_size': 128,
    'max_num_measurements': 16
}
model = LSTMMeasurementPredictor(**model_params)
model.load(model_save_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_rho, test_measurement, _ = test_dataset[124]
rec_rhos, rec_rhos_error, predicted_bases_ids = reconstruct_rho(model, device, test_rho, test_measurement)
pass
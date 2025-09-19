import torch
from src.datasets import MeasurementDataset
from src.data_utils import generate_eigvals
from src.model_utils import reconstruct_from_measurement_predictor
from src.model import LSTMMeasurementPredictor

from torch.utils.data import DataLoader


num_qubits = 2
max_num_measurements = 4**num_qubits

model_input_info = 'full'
eigvals_save_prefix = f'./logs/2qbits/eigvals/full_lstm_measurement_predictor'

model_name = 'full_lstm_measure_basis'
model_save_path = f'./models/{model_name}.pt'

model_params = {
    'num_qubits': num_qubits,
    'layers': 6,
    'hidden_size': 128,
    'max_num_measurements': 4**num_qubits
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTMMeasurementPredictor(**model_params)
model.load(model_save_path)
model.eval()
model.to(device)

batch_size = 64
test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True, num_qubits=num_qubits)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


if model_input_info == 'full':
    input_dim = num_qubits*2*2*2 + 1
elif model_input_info == 'measurement':
    input_dim = 1
elif model_input_info == 'measurement_basis':
    input_dim = num_qubits*2*2*2
    

for num_measurements in range(1, max_num_measurements + 1):
    print(f'Running for {num_measurements} measurements')

    generate_eigvals(
        test_loader,
        reconstruction_fn=lambda measurement, rho: reconstruct_from_measurement_predictor(
            measurement, rho, model, num_measurements - 1),
        save_path=f'{eigvals_save_prefix}_m{num_measurements}.log',
        device=device
    )
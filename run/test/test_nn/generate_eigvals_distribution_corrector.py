import torch
from src.datasets import MeasurementDataset
from src.data_utils import generate_eigvals
from src.model_utils import reconstruct_from_tomography_corrections_predictor
from src.model import TomographyCorrectionsPredictor

from torch.utils.data import DataLoader

def list_to_str(l):
    return '_'.join([str(x) for x in l])


measurement_subsets = [
    [0],
    [8, 12],
    [5, 4, 15],
    [8, 2, 4, 14],
    [13, 6, 11, 15, 2],
    [2, 3, 1, 15, 8, 12],
    [1, 12, 10, 15, 14, 13, 9],
    [13, 10, 7, 4, 0, 3, 14, 8],
    [11, 1, 9, 10, 15, 6, 8, 3, 14],
    [11, 1, 14, 12, 7, 15, 10, 3, 8, 4],
    [15, 14, 2, 12, 10, 8, 9, 5, 0, 13, 7],
    [9, 7, 5, 2, 15, 13, 3, 11, 8, 14, 10, 6],
    [9, 0, 10, 2, 3, 8, 7, 6, 1, 5, 12, 11, 14],
    [2, 8, 9, 3, 11, 7, 5, 14, 1, 13, 12, 15, 0, 6],
    [9, 1, 13, 0, 5, 14, 10, 2, 6, 7, 11, 3, 15, 12, 8],
    [8, 15, 7, 2, 14, 1, 9, 3, 10, 0, 4, 13, 11, 6, 5, 12]
]


num_qubits = 2

model_input_info = 'full'
eigvals_save_prefix = f'./logs/2qbits/eigvals/tomography_corrections_predictor'

batch_size = 64
test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True, num_qubits=num_qubits)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


if model_input_info == 'full':
    input_dim = num_qubits*2*2*2 + 1
elif model_input_info == 'measurement':
    input_dim = 1
elif model_input_info == 'measurement_basis':
    input_dim = num_qubits*2*2*2
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for measurement_subset in measurement_subsets:
    num_measurements = len(measurement_subset)
    print(f'Running for {num_measurements} measurements')

    model_params = {
        'input_dim': num_measurements*input_dim,
        'num_measurements': num_measurements,
        'num_gammas': 4**num_qubits,
        'layers': 6,
        'hidden_size': 64,
    }


    model_name = 'mlp_tomography_corrections_predictor'
    model_name = f'{model_name}_m{list_to_str(measurement_subset)}'
    dir_name = f'tomography_corrections_predictor_m{num_measurements}'
    model_save_path = f'./models/{dir_name}/{model_name}.pt'

    model = TomographyCorrectionsPredictor(**model_params)
    model.load(model_save_path)
    model.eval()
    model.to(device)

    generate_eigvals(
        test_loader,
        reconstruction_fn=lambda measurement, rho: reconstruct_from_tomography_corrections_predictor(
            measurement, model, num_qubits, measurement_subset, model_input_info),
        save_path=f'{eigvals_save_prefix}_m{num_measurements}.log',
        device=device
    )
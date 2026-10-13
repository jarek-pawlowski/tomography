import torch
from src.datasets import MeasurementDataset
from src.data_utils import generate_eigvals
from src.tomography_utils_torch import reconstruct_from_mle

from torch.utils.data import DataLoader


num_qubits = 2
max_num_measurements = 4**num_qubits

eigvals_save_prefix = f'./logs/2qbits/eigvals/mle_reconstruction'


measurement_subsets = [
    # [0],
    # [8, 12],
    # [5, 4, 15],
    # [8, 2, 4, 14],
    # [13, 6, 11, 15, 2],
    # [2, 3, 1, 15, 8, 12],
    # [1, 12, 10, 15, 14, 13, 9],
    # [13, 10, 7, 4, 0, 3, 14, 8],
    # [11, 1, 9, 10, 15, 6, 8, 3, 14],
    # [11, 1, 14, 12, 7, 15, 10, 3, 8, 4],
    # [15, 14, 2, 12, 10, 8, 9, 5, 0, 13, 7],
    # [9, 7, 5, 2, 15, 13, 3, 11, 8, 14, 10, 6],
    # [9, 0, 10, 2, 3, 8, 7, 6, 1, 5, 12, 11, 14],
    # [2, 8, 9, 3, 11, 7, 5, 14, 1, 13, 12, 15, 0, 6],
    # [9, 1, 13, 0, 5, 14, 10, 2, 6, 7, 11, 3, 15, 12, 8],
    [8, 15, 7, 2, 14, 1, 9, 3, 10, 0, 4, 13, 11, 6, 5, 12]
]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


batch_size = 64
test_dataset = MeasurementDataset(root_path='./data/val/', return_density_matrix=True, num_qubits=num_qubits, data_limit=5000)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


for measurement_subset in measurement_subsets:
    num_measurements = len(measurement_subset)
    print(f'Running for {num_measurements} measurements')

    # Reverse subset for MLE (as it takes noisy measurement ids as input)
    measurements_subset_complement = list(set(range(max_num_measurements)) - set(measurement_subset))

    generate_eigvals(
        test_loader,
        reconstruction_fn=lambda measurement, rho: reconstruct_from_mle(
            measurement, measurements_subset=measurements_subset_complement, noise_variance=1.),
        save_path=f'{eigvals_save_prefix}_m{num_measurements}.log',
        device=device
    )
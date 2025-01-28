import sys
sys.path.append('./')

import os 

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.tomography_utils_numpy import Kwiat
from src.datasets import DerandomizedTestMeasurementDataset
from src.model import LSTMDiscreteMeasurementSelector, LSTMDiscreteMeasurementSelectorOptimized, LSTMMeasurementPredictor, LSTMMeasurementSelector
from src.model_utils import collect_rhos_with_closest_kwiat_bases, collect_kwiat_measurements_basis_probabilities_from_discrete_model, collect_measurements_outputs_from_model
from src.logging import plot_matrices

batch_size = 128
data_name = 'Xs'
test_dataset = DerandomizedTestMeasurementDataset(root_path=f'./data/derandomized_test/{data_name}')

basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Kwiat.basis]

# create model
model_name = 'Xs_discrete_lstm_basis_selector_unique_kwiat_basis_cross_entropy_loss_10_noisy_epochs'
# model_name = 'full_lstm_measure_basis'
model_save_path = f'./models/2qbits/{model_name}.pt'

model_params = {
    'num_qubits': 2,
    'possible_basis_matrices': basis_matrices, # 'Kwiat' basis matrices
    'layers': 6,
    'hidden_size': 128,
    'max_num_measurements': 16
}

model = LSTMDiscreteMeasurementSelector(**model_params)
# model = LSTMDiscreteMeasurementSelectorOptimized(**model_params)
# model = LSTMMeasurementPredictor(**model_params)
model.load(model_save_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(len(test_dataset)):
    test_rho, test_measurement = test_dataset[i]
    most_important_measurements_order = torch.argsort(test_measurement, descending=True)
    # if the most important measurements are the ones with the highest values, cant we just check if the value of the measurement is the highest in the first proposed measurements?
    predicted_basis_probabilities, predicted_rhos = collect_kwiat_measurements_basis_probabilities_from_discrete_model(model, device, test_measurement, test_rho)
    sorted_predicted_measurements_ids = torch.argsort(predicted_basis_probabilities, descending=True, dim=1)
    chosen_measurements = []
    for sorted_predicted_measurement in sorted_predicted_measurements_ids:
        for measurement_id in sorted_predicted_measurement:
            if measurement_id not in chosen_measurements:
                chosen_measurements.append(measurement_id)
                break
    measurements_ids = torch.tensor(chosen_measurements)
    # measurements_ids = torch.argmax(predicted_basis_probabilities, dim=1)
    
    predicted_measurements = test_measurement[measurements_ids.cpu()]
    
    # predicted_bases, predicted_measurements, predicted_rhos = collect_measurements_outputs_from_model(model, device, test_measurement, test_rho)
    # measurements_ids = [
    #     f'I:{b[0, 0, 0].real:.2f}-{b[0, 0, 1]:.2f}\nII:{b[1, 0, 0].real:.2f}-{b[1, 0, 1]:.2f}'
    # for b in predicted_bases
    # ]

    # predicted_base_save_path = f'./data/predicted_bases/{data_name}/{model_name}'
    # os.makedirs(predicted_base_save_path, exist_ok=True)
    # np.save(f'{predicted_base_save_path}/predicted_bases_for_state_{i}.npy', predicted_bases.cpu().detach().numpy())

    error = (predicted_rhos.cpu() - test_rho.unsqueeze(0).expand(predicted_rhos.shape)).abs().mean(dim=(1, 2, 3)).detach().numpy()

    # plot bar plot with the values of test measurement in order specified by measurements_ids
    fig, ax = plt.subplots(figsize=(10, 10))
    ax2 = ax.twinx()    
    ax.bar(range(16), predicted_measurements.cpu().detach().numpy(), color='xkcd:blue')
    # set custom bar labels over bars
    for j, value in enumerate(predicted_measurements.cpu().detach().numpy()):
        measurement_id = measurements_ids[j]
        ax.text(j, value + 0.01, f'{measurement_id}', ha='center', va='bottom', color='xkcd:dark blue', rotation=90)

    ax.set_ylabel('measurement value', color='xkcd:dark blue')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis='y', labelcolor='xkcd:dark blue')
    
    ax2.plot(range(16), error, color='xkcd:dark red')
    ax2.set_ylabel('MAE Error', color='xkcd:dark red')
    ax2.tick_params(axis='y', labelcolor='xkcd:dark red')
    ax2.set_ylim(0, 0.12)
    plt.xlabel('measurement index')
    plt.title(f'Test measurement {i} values ordered by most important predicted measurements')
    save_dir = f'./plots/measurement_importance_ordered/{data_name}/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/m_{i}.png')
    plt.close()
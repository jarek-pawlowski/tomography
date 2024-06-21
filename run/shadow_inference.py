import sys
sys.path.append('./')
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset, MeasurementVectorDataset
from src.model_shadow import LSTMMeasurementSelector
from src.torch_utils import train_measurement_predictor, test_measurement_predictor, bases_loss
from src.logging import log_metrics_to_file, plot_metrics_from_file
from src.utils_measure import Pauli, Pauli_c, Pauli_vector



def main():
        
    # Load the saved model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    num_qubits = 6
    batch_size = 1
    numer_of_snapshots = 50
    epochs = 120
    
    basis_matrices = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli.basis]
    basis_matrices_c = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli_c.basis] #complementary basis 
    basis_reconstruction = [torch.tensor(basis, dtype=torch.complex64) for basis in Pauli_vector] #3 normal Pauli matrices
    
    
    # Initialize the model
    model_params = {
        'num_qubits': num_qubits,
        'basis_matrices': basis_matrices, # 'Pauli' basis matrices
        'basis_matrices_c': basis_matrices_c, # 'Pauli complementary' basis matrices
        'basis_reconstruction': basis_reconstruction,
        'layers': 6,
        'hidden_size': 128,
        'max_num_snapshots': numer_of_snapshots,
        'device': device
    }
    model = LSTMMeasurementSelector(**model_params)

    model_save_path = './models/20_JUN_J_spectrum_run_thesis_normal_states_6_qubits_50_shadow_epoch_120.pt'
    
    state_dict = torch.load(model_save_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    # Load the data you want to make predictions on
    root_path = f'./training_states/training_states_{num_qubits}/'
    inference_dataset = MeasurementVectorDataset(num_qubits, root_path=root_path, return_density_matrix=True)
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

    # Iterate over the data and make predictions
    for data in inference_loader:
        inputs = data[0].to(device)
    
        with torch.no_grad():  # No need to track the gradients
            predicted_bases = model(inputs)
        print(predicted_bases)

# Call the function
if __name__ == '__main__':
    main()
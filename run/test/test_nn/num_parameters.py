import os

from src.model_optimized import CombinedLSTMMeasurementPredictor
from src.model_optimized import LSTMReconstructor
from src.model import TomographyCorrectionsPredictor

save_path = './logs/num_parameters/num_parameters.txt'
os.makedirs(os.path.dirname(save_path), exist_ok=True)


for num_qubits in [2, 3, 4]:
    lstm_hs = 256 if num_qubits in [2, 3] else 1024
    n_layers = 1 if num_qubits == 2 else 2

    lstm_adjusted_model_params = {
        'num_qubits': num_qubits,
        'hidden_size': lstm_hs,
        'max_num_measurements': 4**num_qubits,
        'selection_measurements': True,
        'num_layers': n_layers,
    }

    lstm_adjusted = CombinedLSTMMeasurementPredictor(**lstm_adjusted_model_params)

    lstm_random_params = {
        'num_qubits': num_qubits,
        'hidden_size': 256,
        'max_num_measurements': 4**num_qubits,
        'num_layers': n_layers,
    }

    lstm_random = LSTMReconstructor(**lstm_random_params)

    input_dim = num_qubits*2*2*2 + 1
    tomo_corrections_params = {
        'input_dim': 4**num_qubits*input_dim,
        'num_measurements': 4**num_qubits,
        'num_gammas': 4**num_qubits,
        'layers': 6,
        'hidden_size': 64, # 64
    }

    corrector = TomographyCorrectionsPredictor(**tomo_corrections_params)

    models = {
        'combined_lstm_memory': lstm_adjusted,
        'random_lstm': lstm_random,
        'tomography_corrections': corrector,
    }

    with open(save_path, 'a') as f:
        f.write(f'Number of parameters for {num_qubits} qubits:\n')
        for model_name, model in models.items():
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f'{model_name}: {num_params}\n')
        f.write('\n')

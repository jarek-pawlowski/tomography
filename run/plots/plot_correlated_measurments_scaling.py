import sys

from src.test_model import test_measurement_predictor
from src.train import train_measurement_predictor
sys.path.append('./')
import os

import numpy as np
import matplotlib.pyplot as plt


from src.datasets import MeasurementDataset
from src.model import SequentialMeasurementPredictor, LSTMMeasurementPredictor
from src.criterions import torch_bures_distance
from src.logging import load_metrics_from_file


def main():
    num_qubits = 3
    # set paths 
    log_path_lstm = f'./logs/{num_qubits}qbits/full_lstm_measure_basis_meauremnt_dependence.log'
    log_path_pinv_gammas = f'./logs/{num_qubits}qbits/density_matrix_reconstructor_from_pinv_gammas.log'
    
    plot_path = './plots/3_qbits_correlated_measurements_error_bures.png'

    metric_to_plot = 'bures_distance' 
    # metric_to_plot =  'test_loss'
    fixed_metric_name = 'bures_distance' 
    # fixed_metric_name = 'test_mse_loss'
    new_metric_name =  'bures_distance_avg' 
    # new_metric_name = 'test_loss_avg'
    pinv_metrics_name = 'bures_distance_avg' 
    # pinv_metrics_name = 'mse_loss_avg'


    # load data
    metrics_lstm = load_metrics_from_file(log_path_lstm)
    metrics_pinv_gammas = load_metrics_from_file(log_path_pinv_gammas)
    
    num_colors = 2
    cm = plt.get_cmap('tab20')
    fig, ax = plt.subplots()
    # ax.set_prop_cycle(color=[cm(10.*i/num_colors) for i in range(num_colors)])
    # plot
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_pinv_gammas[pinv_metrics_name], label='Tomography with pseudoinverse')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm[metric_to_plot], label='Arbitrary basis LSTM')

    # plt.xticks(np.arange(1, 4**num_qubits+1))
    plt.title('Bures distance for reconstructed density matrix')
    plt.xlabel('Number of measurements')
    plt.ylabel('Bures distance') 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight')

if __name__ == '__main__':
    main()
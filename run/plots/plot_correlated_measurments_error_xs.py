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
from src.log import load_metrics_from_file


def main():
    num_qubits = 3
    markevery = 4
    # set paths 
    log_path_lstm_random_selection = f'./logs/{num_qubits}qbits/xs_lstm_reconstructor_optimized.log'
    log_path_pinv_gammas = f'./logs/{num_qubits}qbits/xs_density_matrix_reconstructor_from_pinv_gammas.log'
    # log_path_discrete_lstm = f'logs/{num_qubits}qbits/xs_combined_lstm_hs256_semi_rand_measurement_predictor_with_measurements_measure_memory_v2_measurement_dependence.log'
    # log_path_discrete_lstm = f'logs/{num_qubits}qbits/xs_combined_lstm_discrete_unique_measurement_selector_measure_memory_measurement_dependence_avg.log'
    log_path_discrete_lstm = f'logs/{num_qubits}qbits/xs_discrete_optimized-v2_lstm2_basis_selector_unique_kwiat_basis_cross_entropy_loss_5_noisy_epochs_lr_decreased_measurement_dependence.log'
    log_path_combined_lstm =  f'logs/{num_qubits}qbits/xs_combined_lstm2_hs256_semi_rand_measurement_predictor_with_measurements_measure_memory_measurement_dependence_avg.log'

    plot_path = f'./plots/{num_qubits}qbits_xs_correlated_measurements_error_bures_log.png'

    metric_to_plot = 'bures_distance' 
    # metric_to_plot =  'test_loss'
    # fixed_metric_name = 'test_mse_loss'
    new_metric_name =  'bures_distance_avg' 
    # new_metric_name = 'test_loss_avg'
    pinv_metrics_name = 'bures_distance_avg' 
    # pinv_metrics_name = 'mse_loss_avg'


    # load data
    metrics_random_lstm = load_metrics_from_file(log_path_lstm_random_selection)
    metrics_pinv_gammas = load_metrics_from_file(log_path_pinv_gammas)
    metrics_combined_lstm = load_metrics_from_file(log_path_combined_lstm)
    metrics_discrete_lstm = load_metrics_from_file(log_path_discrete_lstm)

    # add metric for all correct measurements in tomography
    # tomography_fixed_metrics = np.insert(metrics_tomography[fixed_metric_name], 0, 0)


    plt.rcParams.update({'font.size': 12})
    
    plt.plot(metrics_pinv_gammas['num_measurements'], metrics_pinv_gammas[pinv_metrics_name], color='b', marker='h', markersize=5, markevery=markevery, fillstyle='none', linestyle='-.', label='Tomography with\npseudoinverse')
    plt.plot(metrics_random_lstm['num_measurements'], metrics_random_lstm[new_metric_name], color='m', marker='*', markersize=5, markevery=markevery, linestyle='--', fillstyle='none', label='LSTM with random basis')
    plt.plot(metrics_combined_lstm['num_measurements'], metrics_combined_lstm[metric_to_plot], color='r', marker='s', markersize=5, markevery=markevery, linestyle=':', fillstyle='none', label='LSTM with adjusted basis')
    plt.plot(metrics_discrete_lstm['num_measurements'], metrics_discrete_lstm[metric_to_plot], color='darkred', marker='^', markersize=5, fillstyle='none', linestyle=(0, (1, 3)), label='LSTM with James et al. basis') # noise turned off after 10 epochs


    plt.xticks(np.arange(1, 4**num_qubits+1, 2*markevery))
    plt.yscale('log')
    plt.ylim(1e-4, 3)
    # plt.title('Bures distance for reconstructed density matrix')
    plt.xlabel('Number of measurement outcomes')
    plt.ylabel('Bures distance') 
    plt.legend(prop={'size': 5}) #bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight', dpi=1000)

if __name__ == '__main__':
    main()
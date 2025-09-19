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
    num_qubits = 4
    # set paths 
    log_path_pinv_gammas = f'./logs/{num_qubits}qbits/density_matrix_reconstructor_from_pinv_gammas.log'
    log_path_corections_predictor = f'./logs/{num_qubits}qbits/tomography_corrections_predictor_hs64.log'
    log_path_corections_hs_1024_predictor = f'./logs/{num_qubits}qbits/tomography_corrections_predictor_hs1024.log'
    log_path_lstm_hs1024_random_basis = f'./logs/{num_qubits}qbits/lstm2_reconstructor_optimized.log'
    log_path_lstm_hs1024_random_basis = f'./logs/{num_qubits}qbits/lstm2_hs1024_reconstructor_optimized.log'
    log_path_lstm_2_layers_memory = f'./logs/{num_qubits}qbits/combined_lstm2_semi_rand_measurement_predictor_with_measurements_measure_memory_measurement_dependence.log'
    log_path_lstm_2_layers_hs1024_memory = f'./logs/{num_qubits}qbits/combined_lstm2_hs1024_semi_rand_measurement_predictor_with_measurements_measure_memory_measurement_dependence.log'
    log_path_lstm_2_layers_no_measurements_memory = f'./logs/{num_qubits}qbits/combined_lstm2_discrete_unique_measurement_selector_no_measurements_measure_memory_measurement_dependence.log'

    plot_path = './plots/4_qbits_correlated_measurements_error_bures_log.png'

    metric_to_plot = 'bures_distance' 
    # metric_to_plot =  'test_loss'
    fixed_metric_name = 'bures_distance' 
    # fixed_metric_name = 'test_mse_loss'
    new_metric_name =  'bures_distance_avg' 
    # new_metric_name = 'test_loss_avg'
    pinv_metrics_name = 'bures_distance_avg' 
    # pinv_metrics_name = 'mse_loss_avg'

    xaxis_name = 'num_measurements'

    # load data
    metrics_pinv_gammas = load_metrics_from_file(log_path_pinv_gammas)
    metrics_corrections_predictor = load_metrics_from_file(log_path_corections_predictor)
    metrics_corrections_hs_1024_predictor = load_metrics_from_file(log_path_corections_hs_1024_predictor)
    metrics_lstm_random_basis = load_metrics_from_file(log_path_lstm_hs1024_random_basis)
    metrics_lstm_hs1024_random_basis = load_metrics_from_file(log_path_lstm_hs1024_random_basis)
    metrics_lstm_2_layers_memory = load_metrics_from_file(log_path_lstm_2_layers_memory)
    metrics_lstm_2_layers_hs1024_memory = load_metrics_from_file(log_path_lstm_2_layers_hs1024_memory)
    metrics_lstm_2_layers_hs1024_no_measurements_memory = load_metrics_from_file(log_path_lstm_2_layers_no_measurements_memory)

    plt.rcParams.update({'font.size': 12})
    # fig, ax = plt.subplots()
    # ax.set_prop_cycle(color=[cm(10.*i/num_colors) for i in range(num_colors)])
    # plot
    plt.plot(metrics_pinv_gammas[xaxis_name], metrics_pinv_gammas[pinv_metrics_name], color='b', marker='h', markersize=6, markevery=2, fillstyle='none', linestyle='-.', label='Tomography with\npseudoinverse')
    plt.plot(metrics_corrections_predictor[xaxis_name], metrics_corrections_predictor[new_metric_name], color='orange', marker='o', linestyle='-', markersize=6, markevery=2, fillstyle='none', label='Corrector NN')
    # plt.plot(metrics_corrections_hs_1024_predictor[xaxis_name], metrics_corrections_hs_1024_predictor[new_metric_name], color='green', marker='s', linestyle='-', markersize=6, fillstyle='none', label='Corrector NN HS1024')
    # plt.plot(metrics_lstm_hs1024_random_basis[xaxis_name], metrics_lstm_hs1024_random_basis[new_metric_name], color='red', marker='^', linestyle='-', markevery=8, markersize=5, fillstyle='none', label='LSTM 2 layers HS1024 with random basis\n(with memory)')
    # plt.plot(metrics_lstm_2_layers_memory[xaxis_name], metrics_lstm_2_layers_memory[metric_to_plot], color='purple', marker='v', linestyle='-', markevery=8, markersize=5, fillstyle='none', label='LSTM 2 layers with adjusted basis\n(with memory)')
    plt.plot(metrics_lstm_random_basis[xaxis_name], metrics_lstm_random_basis[new_metric_name], color='m', marker='*', linestyle='-', markevery=8, markersize=5, fillstyle='none', label='LSTM with random basis')
    plt.plot(metrics_lstm_2_layers_hs1024_memory[xaxis_name], metrics_lstm_2_layers_hs1024_memory[metric_to_plot], color='r', marker='s', linestyle='-', markevery=8, markersize=5, fillstyle='none', label='LSTM with adjusted basis')
    # plt.plot(metrics_lstm_2_layers_hs1024_no_measurements_memory[xaxis_name], metrics_lstm_2_layers_hs1024_no_measurements_memory[metric_to_plot], color='magenta', marker='*', linestyle='-', markevery=8, markersize=5, fillstyle='none', label='LSTM 2 layers with adjusted basis\n(no measurement input, with memory)')

    # plt.xticks(np.arange(1, 4**num_qubits+1))
    # plt.title('Bures distance for reconstructed density matrix')
    plt.yscale('log')
    plt.ylim(1e-2, 1e-0)
    plt.xlabel('Number of measurement outcomes')
    plt.ylabel('Bures distance') 
    plt.legend(prop={'size': 9}, loc='lower left') #bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight', dpi=1000)

if __name__ == '__main__':
    main()
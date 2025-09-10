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
    # set paths 
    log_path_lstm = f'./logs/{num_qubits}qbits/full_lstm_measure_basis_meauremnt_dependence.log'
    log_path_lstm_random = f'./logs/{num_qubits}qbits/lstm_reconstructor.log'
    log_path_lstm_random_memory = f'./logs/{num_qubits}qbits/lstm_reconstructor_optimized.log'
    log_path_pinv_gammas = f'./logs/{num_qubits}qbits/density_matrix_reconstructor_from_pinv_gammas.log'
    log_path_corections_predictor = f'./logs/{num_qubits}qbits/tomography_corrections_predictor.log'
    log_path_corections_predictor_hs1024 = f'./logs/{num_qubits}qbits/tomography_corrections_predictor_hs1024.log'
    log_path_stack_lstm = f'./logs/{num_qubits}qbits/full_lstm_measure_basis_stacked_input+hs256_no_bases_loss_meauremnt_dependence.log'
    log_path_stack_1024_lstm = f'./logs/{num_qubits}qbits/full_lstm_measure_basis_stacked_input-hs1024_no_bases_loss_measurement_dependence.log'
    log_path_discrete_no_measurements_lstm = f'./logs/{num_qubits}qbits/discrete_no_measurements_lstm2_basis_selector_unique_kwiat_basis_cross_entropy_loss_5_noisy_epochs_lr_decreased_measurement_dependence.log'
    log_path_combined_lstm_no_measurements = f'./logs/{num_qubits}qbits/combined_lstm_discrete_unique_measurement_selector_no_measurements_soft_temp_1_5_measurement_dependence.log'
    log_path_combined_soft_lstm_no_measurements_memory = f'./logs/{num_qubits}qbits/combined_lstm_discrete_unique_measurement_selector_no_measurements_soft_measure_memory_measurement_dependence.log'
    log_path_combined_lstm_no_measurements_memory = f'./logs/{num_qubits}qbits/combined_lstm_discrete_unique_measurement_selector_no_measurements_measure_memory_measurement_dependence.log'
    log_path_lstm_memory_no_measurements = f'./logs/{num_qubits}qbits/combined_lstm_measurement_predictor_no_measurements_measure_memory_measurement_dependence.log'
    log_path_lstm_memory_no_measurements_v2 = f'./logs/{num_qubits}qbits/combined_lstm_measurement_predictor_no_measurments_measure_memory_v2_measurement_dependence.log'
    log_path_lstm_memory = f'./logs/{num_qubits}qbits/combined_lstm_measurement_predictor_measure_memory_measurement_dependence.log'


    plot_path = './plots/3_qbits_correlated_measurements_error_bures_log.png'

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
    metrics_stack_lstm = load_metrics_from_file(log_path_stack_lstm)
    metrics_stack_1024_lstm = load_metrics_from_file(log_path_stack_1024_lstm)
    metrics_lstm_random = load_metrics_from_file(log_path_lstm_random)
    metrics_lstm_random_memory = load_metrics_from_file(log_path_lstm_random_memory)
    metrics_pinv_gammas = load_metrics_from_file(log_path_pinv_gammas)
    metrics_corrections_predictor = load_metrics_from_file(log_path_corections_predictor)
    metrics_corrections_predictor_hs1024 = load_metrics_from_file(log_path_corections_predictor_hs1024)
    metrics_discrete_no_measurements_lstm = load_metrics_from_file(log_path_discrete_no_measurements_lstm)
    metrics_lstm_combined_temp = load_metrics_from_file(log_path_combined_lstm_no_measurements)
    metrics_lstm_combined_soft_memory = load_metrics_from_file(log_path_combined_soft_lstm_no_measurements_memory)
    metrics_lstm_combined_memory = load_metrics_from_file(log_path_combined_lstm_no_measurements_memory)
    metrics_lstm_memory_no_measurements = load_metrics_from_file(log_path_lstm_memory_no_measurements)
    metrics_lstm_memory_no_measurements_v2 = load_metrics_from_file(log_path_lstm_memory_no_measurements_v2)
    metrics_lstm_memory = load_metrics_from_file(log_path_lstm_memory)

    plt.rcParams.update({'font.size': 12})
    # fig, ax = plt.subplots()
    # ax.set_prop_cycle(color=[cm(10.*i/num_colors) for i in range(num_colors)])
    # plot
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_pinv_gammas[pinv_metrics_name], color='b', marker='h', markevery=2, markersize=6, fillstyle='none', linestyle='-.', label='Tomography with\npseudoinverse')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_corrections_predictor[new_metric_name], color='orange', marker='o', markevery=2, linestyle='-', markersize=6, fillstyle='none', label='Corrector NN')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_corrections_predictor_hs1024[new_metric_name], color='g', marker='^', markevery=2, linestyle='--', markersize=6, fillstyle='none', label='Corrector NN hs1024')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm[metric_to_plot], color='r', marker='s', markevery=2, markersize=5, linestyle=':', fillstyle='none', label='Original LSTM with\nadjusted basis')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_stack_lstm[metric_to_plot], color='g', marker='x', markevery=2, markersize=5, linestyle='--', fillstyle='none', label='Stacked LSTM with\nadjusted basis')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_stack_1024_lstm[metric_to_plot], color='r', marker='s', markevery=2, markersize=5, linestyle=':', fillstyle='none', label='LSTM with\nadjusted basis (hs1024)')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm_random[new_metric_name], color='m', marker='*', markevery=2, markersize=5, linestyle='--', fillstyle='none', label='LSTM with random basis')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm_random_memory[new_metric_name], color='tab:blue', marker='P', markevery=2, markersize=5, linestyle=':', fillstyle='none', label='LSTM with random basis\n(with memory)')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_discrete_no_measurements_lstm[metric_to_plot], color='purple', marker='h', markevery=2, linestyle=(0, (3, 6)), markersize=5, fillstyle='none', label='LSTM with James et al. basis\n(basis only)')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm_combined_temp[metric_to_plot], color='c', marker='D', markevery=2, linestyle='-.', markersize=5, fillstyle='none', label='LSTM with James et al. basis\n(soft, basis only, t=1.5)')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm_combined_soft_memory[metric_to_plot], color='tab:gray', marker='X', markevery=2, linestyle='-', markersize=5, fillstyle='none', label='LSTM with James et al. basis\n(soft, basis only, with memory)')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm_combined_memory[metric_to_plot], color='y', marker='v', markevery=2, linestyle='--', markersize=5, fillstyle='none', label='LSTM with James et al. basis\n(basis only, with memory)')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm_memory_no_measurements[metric_to_plot], color='tab:olive', marker='p', markevery=2, linestyle=':', markersize=5, fillstyle='none', label='LSTM with adjusted basis\n(basis only, with memory)')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm_memory_no_measurements_v2[metric_to_plot], color='tab:orange', marker='8', markevery=2, linestyle='--', markersize=5, fillstyle='none', label='LSTM with adjusted basis\n(basis only, with memory) v2')
    plt.plot(np.arange(1, 4**num_qubits+1), metrics_lstm_memory[metric_to_plot], color='tab:brown', marker='8', markevery=2, linestyle='-.', markersize=5, fillstyle='none', label='LSTM with adjusted basis\n(with memory)')

    # plt.xticks(np.arange(1, 4**num_qubits+1))
    # plt.title('Bures distance for reconstructed density matrix')
    plt.yscale('log')
    plt.ylim(1e-2, 1e-0)
    plt.xlabel('Number of measurement outcomes')
    plt.ylabel('Bures distance') 
    plt.legend(prop={'size': 5}) #bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight', dpi=1000)

if __name__ == '__main__':
    main()
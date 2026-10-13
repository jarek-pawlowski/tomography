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
    log_path_ref_lstm_random_2qbits = './logs/2qbits/lstm_reconstructor.log'
    log_path_ref_lstm_random_3qbits = './logs/3qbits/lstm2_reconstructor_optimized.log'
    log_path_ref_lstm_random_4qbits = './logs/4qbits/lstm2_hs1024_reconstructor_optimized.log'
    
    log_path_lstm_random_2qbits = './logs/2qbits/xs_lstm_reconstructor_optimized.log'
    log_path_lstm_random_3qbits = './logs/3qbits/xs_lstm_reconstructor_optimized.log'
    log_path_lstm_random_4qbits = './logs/4qbits/xs_lstm_reconstructor_optimized.log'


    metric_to_plot = 'bures_distance' 
    # metric_to_plot =  'test_loss'
    # fixed_metric_name = 'test_mse_loss'
    new_metric_name =  'bures_distance_avg' 
    # new_metric_name = 'test_loss_avg'
    pinv_metrics_name = 'bures_distance_avg' 
    # pinv_metrics_name = 'mse_loss_avg'

    plot_path = './plots/xs_correlated_measurements_scaling_error_bures_log.png'


    # load data
    metrics_ref_lstm_random_2qbits = load_metrics_from_file(log_path_ref_lstm_random_2qbits)
    metrics_ref_lstm_random_3qbits = load_metrics_from_file(log_path_ref_lstm_random_3qbits)
    metrics_ref_lstm_random_4qbits = load_metrics_from_file(log_path_ref_lstm_random_4qbits)
    metrics_lstm_random_2qbits = load_metrics_from_file(log_path_lstm_random_2qbits)
    metrics_lstm_random_3qbits = load_metrics_from_file(log_path_lstm_random_3qbits)
    metrics_lstm_random_4qbits = load_metrics_from_file(log_path_lstm_random_4qbits)
    # add metric for all correct measurements in tomography
    # tomography_fixed_metrics = np.insert(metrics_tomography[fixed_metric_name], 0, 0)


    plt.rcParams.update({'font.size': 12})
    
    plt.plot(metrics_ref_lstm_random_2qbits['num_measurements'] / max(metrics_ref_lstm_random_2qbits['num_measurements']), metrics_ref_lstm_random_2qbits[pinv_metrics_name], color='royalblue', marker='v', markersize=5, markevery=1, fillstyle='none', linestyle='-', label='2 qubits')
    plt.plot(metrics_lstm_random_2qbits['num_measurements'] / max(metrics_lstm_random_2qbits['num_measurements']), metrics_lstm_random_2qbits[new_metric_name], color='darkblue', marker='v', markersize=5, markevery=1, linestyle='--', fillstyle='none', label='2 qubits\n(X states)')
    plt.plot(metrics_ref_lstm_random_3qbits['num_measurements'] / max(metrics_ref_lstm_random_3qbits['num_measurements']), metrics_ref_lstm_random_3qbits[pinv_metrics_name], color='limegreen', marker='o', markersize=5, markevery=4, fillstyle='none', linestyle='-', label='3 qubits')
    plt.plot(metrics_lstm_random_3qbits['num_measurements'] / max(metrics_lstm_random_3qbits['num_measurements']), metrics_lstm_random_3qbits[new_metric_name], color='darkgreen', marker='o', markersize=5, markevery=4, linestyle='--', fillstyle='none', label='3 qubits\n(X states)')
    plt.plot(metrics_ref_lstm_random_4qbits['num_measurements'] / max(metrics_ref_lstm_random_4qbits['num_measurements']), metrics_ref_lstm_random_4qbits[pinv_metrics_name], color='tomato', marker='D', markersize=5, markevery=16, fillstyle='none', linestyle='-', label='4 qubits')
    plt.plot(metrics_lstm_random_4qbits['num_measurements'] / max(metrics_lstm_random_4qbits['num_measurements']), metrics_lstm_random_4qbits[new_metric_name], color='crimson', marker='D', markersize=5, markevery=16, linestyle='--', fillstyle='none', label='4 qubits\n(X states)')

    plt.yscale('log')
    plt.ylim(0.0015, 1.)
    # plt.title('Bures distance for reconstructed density matrix')
    plt.xlabel(r'$M/M_{\max}$')
    plt.ylabel('Bures distance') 
    plt.legend(prop={'size': 9}, loc='lower left') #bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight', dpi=1000)

if __name__ == '__main__':
    main()
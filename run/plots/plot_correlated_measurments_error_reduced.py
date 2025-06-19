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
    # set paths 
    global_dir = './logs/1qbit/'
    log_path_tomography = f'{global_dir}rho_test_varying_random_measurement_clipped_tomography_avg.log'
    log_path_zeroed_tomography = f'{global_dir}rho_test_varying_zeroed_measurement_clipped_tomography_avg.log'
    log_path_pinv_gammas = f'{global_dir}density_matrix_reconstructor_from_pinv_gammas_v2.log'
    log_path_hlp = f'{global_dir}density_matrix_reconstructor_from_hlp.log'
    log_path_tomography_corrections = f'{global_dir}tomography_corrections_predictor_subset.log'
    log_path_lstm = f'{global_dir}full_lstm_measure_basis_meauremnt_dependence.log'
    log_path_tomography_corrections_basis_only = f'{global_dir}basis_only_tomography_corrections_predictor.log'
    log_path_tomography_corrections_regularization_only = f'{global_dir}simple_tomography_corrections_predictor.log'
    log_path_tomography_corrections_basis_and_reg_only = f'{global_dir}simple_basis_only_tomography_corrections_predictor.log'

    plot_path = './plots/correlated_measurements_1qbit_error_bures_distance_final_seminar.png'

    fixed_metric_name = 'bures_distance' 
    # fixed_metric_name = 'test_mse_loss'
    metrics_name =  'bures_distance_avg' 
    # metrics_name = 'mse_loss_avg'
    corrections_metrics_name = 'bures_distance_avg' 
    # corrections_metrics_name = 'test_loss_avg'
    lstm_metrics_name = 'bures_distance'
    # lstm_metrics_name = 'test_loss'


    # load data
    metrics_tomography = load_metrics_from_file(log_path_tomography)
    metrics_zeroed_tomography = load_metrics_from_file(log_path_zeroed_tomography)
    metrics_pinv_gammas = load_metrics_from_file(log_path_pinv_gammas)
    metrics_hlp = load_metrics_from_file(log_path_hlp)
    metrics_tomography_corrections = load_metrics_from_file(log_path_tomography_corrections)
    metrics_tomography_corrections_basis_only = load_metrics_from_file(log_path_tomography_corrections_basis_only)
    metrics_tomography_corrections_regularization_only = load_metrics_from_file(log_path_tomography_corrections_regularization_only)
    metrics_tomography_corrections_basis_and_reg_only = load_metrics_from_file(log_path_tomography_corrections_basis_and_reg_only)
    metrics_lstm = load_metrics_from_file(log_path_lstm)

    # add metric for all correct measurements in tomography
    tomography_fixed_metrics = np.insert(metrics_tomography[fixed_metric_name], 0, 0)
    tomography_fixed_metrics = np.flip(tomography_fixed_metrics)[1:]

    zeroed_tomography_fixed_metrics = np.insert(metrics_zeroed_tomography[fixed_metric_name], 0, 0)
    zeroed_tomography_fixed_metrics = np.flip(zeroed_tomography_fixed_metrics)[1:]

    plt.rcParams.update({'font.size': 12})
    xaxis = np.arange(1, 5)
    # plot
    # plt.plot(xaxis, tomography_fixed_metrics, label='Kwiat basis tomography\nwith randomized measurements')
    # plt.plot(xaxis, zeroed_tomography_fixed_metrics, label='Kwiat basis tomography\nwith zeroed measurements')
    plt.plot(xaxis, metrics_pinv_gammas[metrics_name], color='b', marker='h', markersize=6, fillstyle='none', linestyle='-.', label='Tomography with\npseudoinverse')
    plt.plot(xaxis, metrics_tomography_corrections[corrections_metrics_name], color='orange', marker='o', linestyle='-', markersize=6, fillstyle='none', label='Corrector NN')
    # plt.plot(xaxis, metrics_tomography_corrections_basis_only[corrections_metrics_name], color='g', marker='x', markersize=5, linestyle='--', label='Corrector NN\n(basis only)')
    plt.plot(xaxis, metrics_lstm[lstm_metrics_name], color='r', marker='s', markersize=5, linestyle=':', fillstyle='none', label='LSTM with\nadjusted basis')
    plt.plot(xaxis, metrics_hlp[metrics_name], color='silver', fillstyle='none', linestyle='-', linewidth=4.5, zorder=-1, label='Analytical\napproximations')
    # plt.plot(xaxis, metrics_tomography_corrections_regularization_only[corrections_metrics_name], label='Tomography corrections predictor\nwith regularization loss only')
    # plt.plot(xaxis, metrics_tomography_corrections_basis_and_reg_only[corrections_metrics_name], label='Tomography corrections predictor\nusing basis input and regularization loss only')

    plt.xticks(np.arange(1, 5))
    # plt.title('Bures distance for reconstructed density matrix')
    plt.xlabel('Number of measurement outcomes')
    plt.ylabel('Bures distance') 
    plt.legend(prop={'size': 10}) #bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight', dpi=1000)

if __name__ == '__main__':
    main()
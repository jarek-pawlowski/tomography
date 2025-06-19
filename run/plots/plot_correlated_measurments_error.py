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
    # set paths 
    log_path_lstm = './logs/full_lstm_measure_basis_meauremnt_dependence.log'
    log_path_lstm_no_selection = './logs/2qbits/full_lstm_measure_basis_no_measurement_for_selection_meauremnt_dependence.log'
    log_path_smp = './logs/smp_measure_basis_meauremnt_dependence.log'
    log_path_kwiat_basis_lin_comb_lstm = './logs/full_lstm_basis_selector_v3_kwiat_basis_loss_meauremnt_dependence.log'
    log_path_lin_comb_lstm = './logs/full_lstm_basis_selector_v3_meauremnt_dependence.log'
    log_path_kwiat_basis_lstm = './logs/full_lstm_measure_basis_kwiat_basis_loss_meauremnt_dependence.log'
    log_path_discrete_kwiat_basis_lstm = './logs/discrete_lstm_basis_selector_reduced_kwiat_basis_cross_entropy_loss_measuremnt_dependence.log'
    log_path_discrete_noise_break_kwiat_basis_lstm = './logs/discrete_lstm_basis_selector_reduced_kwiat_basis_cross_entropy_loss_10_noisy_epochs_measuremnt_dependence.log'
    log_path_discrete_noise_break_unique_kwiat_basis_lstm = './logs/discrete_lstm_basis_selector_unique_kwiat_basis_cross_entropy_loss_10_noisy_epochs_measuremnt_dependence.log'
    log_path_tomography = './logs/2qbit/rho_test_varying_random_measurement_clipped_tomography_avg.log'
    log_path_zeroed_tomography = './logs/2qbit/rho_test_varying_zeroed_measurement_clipped_tomography_avg.log'
    log_path_mle_intensity =  './logs/2qbit/mle_intensity_rho_test_varying_measurement_clipped_tomography_avg.log'
    log_path_mle = './logs/2qbit/mle_rho_test_varying_measurement_clipped_tomography_avg.log'
    log_path_basis_gammas = './logs/density_matrix_reconstructor_from_basis_gammas_measurements_subset.log'
    log_path_pinv_gammas = './logs/2qbit/density_matrix_reconstructor_from_pinv_gammas_v2.log'
    log_path_reconstructor = './logs/density_matrix_reconstructor_measurements_subset.log'
    log_path_tomography_corrections = './logs/tomography_corrections_predictor_subset.log'
    log_path_tomography_corrections_basis_only = './logs/tomography_corrections_predictor_from_measurement_basis.log'
    log_path_m2_tomography_corrections_basis_only = './logs/2qbits/tomography_m2_corrections_predictor_basis_only.log'
    log_path_discrete_measurement_basis_tomography_corrections_lstm = './logs/tomo_corrections_discrete_lstm_basis_selector_unique_kwiat_basis_cross_entropy_loss_measurement_dependence.log'
    log_path_mean_reconstruction = './logs/2qbit/density_matrix_reconstructor_from_mean.log'

    plot_path = './plots/correlated_measurements_error_bures_new_final_seminar.png'

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
    metrics_lstm_no_selection = load_metrics_from_file(log_path_lstm_no_selection)
    metrics_smp = load_metrics_from_file(log_path_smp)
    metrics_kwiat_basis_lstm = load_metrics_from_file(log_path_kwiat_basis_lstm)
    metrics_discrete_kwiat_basis_lstm = load_metrics_from_file(log_path_discrete_kwiat_basis_lstm)
    metrics_discrete_noise_break_kwiat_basis_lstm = load_metrics_from_file(log_path_discrete_noise_break_kwiat_basis_lstm)
    metrics_discrete_noise_break_unique_kwiat_basis_lstm = load_metrics_from_file(log_path_discrete_noise_break_unique_kwiat_basis_lstm)
    metrics_lin_comb_lstm = load_metrics_from_file(log_path_lin_comb_lstm)
    metrics_kwiat_basis_lin_comb_lstm = load_metrics_from_file(log_path_kwiat_basis_lin_comb_lstm)
    metrics_tomography = load_metrics_from_file(log_path_tomography)
    metrics_zeroed_tomography = load_metrics_from_file(log_path_zeroed_tomography)
    metrics_mle_intensity = load_metrics_from_file(log_path_mle_intensity)
    metrics_mle = load_metrics_from_file(log_path_mle)
    metrics_basis_gammas = load_metrics_from_file(log_path_basis_gammas)
    metrics_pinv_gammas = load_metrics_from_file(log_path_pinv_gammas)
    metrics_reconstructor = load_metrics_from_file(log_path_reconstructor)
    metrics_tomography_corrections = load_metrics_from_file(log_path_tomography_corrections)
    metrics_tomography_corrections_basis_only = load_metrics_from_file(log_path_tomography_corrections_basis_only)
    metrics_m2_tomography_corrections_basis_only = load_metrics_from_file(log_path_m2_tomography_corrections_basis_only)
    metrics_discrete_measurement_basis_tomography_corrections_lstm = load_metrics_from_file(log_path_discrete_measurement_basis_tomography_corrections_lstm)
    metrics_mean_reconstruction = load_metrics_from_file(log_path_mean_reconstruction)

    # add metric for all correct measurements in tomography
    # tomography_fixed_metrics = np.insert(metrics_tomography[fixed_metric_name], 0, 0)
    tomography_fixed_metrics = np.flip(metrics_tomography[fixed_metric_name])[1:]

    # zeroed_tomography_fixed_metrics = np.insert(metrics_zeroed_tomography[fixed_metric_name], 0, 0)
    zeroed_tomography_fixed_metrics = np.flip(metrics_zeroed_tomography[fixed_metric_name])

    mle_intensity_fixed_metrics = np.insert(metrics_mle_intensity[fixed_metric_name], 0, 0)
    mle_intensity_fixed_metrics = np.flip(mle_intensity_fixed_metrics)

    mle_fixed_metrics = np.insert(metrics_mle[fixed_metric_name], 0, 0)
    mle_fixed_metrics = np.flip(mle_fixed_metrics)

    metrics_mean_reconstruction_expanded = np.repeat(metrics_mean_reconstruction[fixed_metric_name], 16)
    
    plt.rcParams.update({'font.size': 12})
    
    # num_colors = 20
    # cm = plt.get_cmap('tab20')
    # fig, ax = plt.subplots()
    # ax.set_prop_cycle(color=[cm(1.*i/num_colors) for i in range(num_colors)])
    # plot
    # plt.plot(np.arange(1, 17), tomography_fixed_metrics, label='Kwiat basis tomography')
    # plt.plot(np.arange(1, 17), zeroed_tomography_fixed_metrics, label='Kwiat basis tomography with zeroed measurements')
    plt.plot(np.arange(1, 17), metrics_pinv_gammas[pinv_metrics_name], color='b', marker='h', markersize=5, fillstyle='none', linestyle='-.', label='Tomography with\npseudoinverse')
    plt.plot(np.arange(1, 17), mle_fixed_metrics, color='k', marker='D', markersize=5, fillstyle='none', linestyle=((0, (5, 1, 2, 1))), label='MLE')
    plt.plot(np.arange(1, 17), mle_intensity_fixed_metrics, color='c', marker='v', markersize=5, fillstyle='none', linestyle=((0, (3, 4))), label='MLE (with intensity)')
    # plt.plot(np.arange(1, 17), metrics_basis_gammas[new_metric_name], label='Tomography with measurement projector gammas')
    # plt.plot(np.arange(1, 17), metrics_reconstructor[new_metric_name], label='Fully connected NN reconstructor\non random measurements')
    # plt.plot(np.arange(1, 17), metrics_smp[metric_to_plot], label='Arbitrary basis fully connected NN')
    plt.plot(np.arange(1, 17), metrics_tomography_corrections[new_metric_name], color='orange', marker='o', linestyle='-', markersize=5, fillstyle='none', label='Corrector NN')
    # plt.plot(np.arange(1, 17), metrics_tomography_corrections_basis_only[new_metric_name], color='g', marker='x', markersize=5, linestyle='--', label='Corrector NN (basis only)')
    # plt.plot(np.arange(1, 17), metrics_m2_tomography_corrections_basis_only[new_metric_name], color='lime', marker='|', markersize=5, linestyle=(0, (3, 3)), label='$M^2$-Corrector NN (basis only)')
    # plt.plot(np.arange(1, 17), metrics_discrete_noise_break_unique_kwiat_basis_lstm[metric_to_plot], color='darkred', marker='^', markersize=5, fillstyle='none', linestyle=(0, (1, 3)), label='LSTM with James et al. basis') # noise turned off after 10 epochs
    plt.plot(np.arange(1, 17), metrics_lstm[metric_to_plot], color='r', marker='s', markersize=5, linestyle=':', fillstyle='none', label='LSTM with adjusted basis')
    # plt.plot(np.arange(1, 17), metrics_lstm_no_selection[metric_to_plot], color='magenta', marker='p', markersize=5, fillstyle='none', linestyle=(0, (2, 5)), label='LSTM with adjusted basis\n(basis only)')
    # plt.plot(np.arange(1, 17), metrics_kwiat_basis_lstm[metric_to_plot], label='Arbitrary basis LSTM with Kwiat basis loss')
    # plt.plot(np.arange(1, 17), metrics_lin_comb_lstm[metric_to_plot], label='LSTM with linear combination of Kwiat basis')
    # plt.plot(np.arange(1, 17), metrics_kwiat_basis_lin_comb_lstm[metric_to_plot], label='LSTM with linear combination of Kwiat basis and loss')
    # plt.plot(np.arange(1, 17), metrics_discrete_kwiat_basis_lstm[metric_to_plot], label='LSTM from discrete Kwiat basis')
    # plt.plot(np.arange(1, 17), metrics_discrete_noise_break_kwiat_basis_lstm[metric_to_plot], label='LSTM from discrete Kwiat basis, noise turned off after 10 epochs')

    # plt.plot(np.arange(1, 17), metrics_discrete_measurement_basis_tomography_corrections_lstm[fixed_metric_name], label='Tomography corrections LSTM predictor from discrete unique Kwiat basis')
    # plt.plot(np.arange(1, 17), metrics_mean_reconstruction_expanded, '--', label='Mean reconstruction')

    plt.xticks(np.arange(1, 17))
    # plt.yscale('log')
    # plt.ylim(1e-5, 3)
    # plt.title('Bures distance for reconstructed density matrix')
    plt.xlabel('Number of measurement outcomes')
    plt.ylabel('Bures distance') 
    plt.legend(prop={'size': 10}) #bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight', dpi=1000)

if __name__ == '__main__':
    main()
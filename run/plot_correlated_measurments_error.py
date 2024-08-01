import sys
sys.path.append('./')
import os

import numpy as np
import matplotlib.pyplot as plt


from src.datasets import MeasurementDataset
from src.model import SequentialMeasurementPredictor, LSTMMeasurementPredictor
from src.torch_utils import train_measurement_predictor, test_measurement_predictor, torch_bures_distance
from src.logging import load_metrics_from_file


def main():
    # set paths 
    log_path_lstm = './logs/full_lstm_measure_basis_meauremnt_dependence.log'
    log_path_smp = './logs/smp_measure_basis_meauremnt_dependence.log'
    log_path_kwiat_basis_lin_comb_lstm = './logs/full_lstm_basis_selector_v3_kwiat_basis_loss_meauremnt_dependence.log'
    log_path_lin_comb_lstm = './logs/full_lstm_basis_selector_v3_meauremnt_dependence.log'
    log_path_kwiat_basis_lstm = './logs/full_lstm_measure_basis_kwiat_basis_loss_meauremnt_dependence.log'
    log_path_discrete_kwiat_basis_lstm = './logs/discrete_lstm_basis_selector_reduced_kwiat_basis_cross_entropy_loss_measuremnt_dependence.log'
    log_path_discrete_noise_break_kwiat_basis_lstm = './logs/discrete_lstm_basis_selector_reduced_kwiat_basis_cross_entropy_loss_10_noisy_epochs_measuremnt_dependence.log'
    log_path_discrete_noise_break_unique_kwiat_basis_lstm = './logs/discrete_lstm_basis_selector_unique_kwiat_basis_cross_entropy_loss_10_noisy_epochs_measuremnt_dependence.log'
    log_path_tomography = './logs/2qbit/rho_test_varying_random_measurement_clipped_tomography_avg.log'
    log_path_zeroed_tomography = './logs/2qbit/rho_test_varying_zeroed_measurement_clipped_tomography_avg.log'
    log_path_mle_intensity = './logs/rho_varying_multiple_measurements/rho_test_varying_measurement_clipped_optimized_intensity.log'
    log_path_mle = './logs/rho_varying_multiple_measurements/rho_test_varying_measurement_clipped_optimized.log'
    log_path_basis_gammas = './logs/density_matrix_reconstructor_from_basis_gammas_measurements_subset.log'
    log_path_pinv_gammas = './logs/density_matrix_reconstructor_from_pinv_gammas_measurements_subset_numerics_100.log'
    log_path_reconstructor = './logs/density_matrix_reconstructor_measurements_subset.log'
    log_path_tomography_corrections = f'./logs/tomography_corrections_predictor_subset.log'

    plot_path = './plots/correlated_measurements_error_mse_new_nn_only.png'

    metric_to_plot = 'test_loss'
    fixed_metric_name = 'test_mse_loss'

    # load data
    metrics_lstm = load_metrics_from_file(log_path_lstm)
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

    # add metric for all correct measurements in tomography
    # tomography_fixed_metrics = np.insert(metrics_tomography[fixed_metric_name], 0, 0)
    tomography_fixed_metrics = np.flip(metrics_tomography[fixed_metric_name])[1:]

    # zeroed_tomography_fixed_metrics = np.insert(metrics_zeroed_tomography[fixed_metric_name], 0, 0)
    zeroed_tomography_fixed_metrics = np.flip(metrics_zeroed_tomography[fixed_metric_name])

    mle_intensity_fixed_metrics = np.insert(metrics_mle_intensity[fixed_metric_name], 0, 0)
    mle_intensity_fixed_metrics = np.flip(mle_intensity_fixed_metrics)[1:]

    mle_fixed_metrics = np.insert(metrics_mle[fixed_metric_name], 0, 0)
    mle_fixed_metrics = np.flip(mle_fixed_metrics)[1:]

    # plot
    plt.plot(np.arange(1, 17), metrics_reconstructor['test_loss_avg'], label='Fully connected NN reconstructor on random measurements')
    plt.plot(np.arange(1, 17), metrics_smp[metric_to_plot], label='Arbitrary basis fully connected NN')
    plt.plot(np.arange(1, 17), metrics_lstm[metric_to_plot], label='Arbitrary basis LSTM')
    plt.plot(np.arange(1, 17), metrics_kwiat_basis_lstm[metric_to_plot], label='Arbitrary basis LSTM with Kwiat basis loss')
    plt.plot(np.arange(1, 17), metrics_lin_comb_lstm[metric_to_plot], label='LSTM with linear combination of Kwiat basis')
    plt.plot(np.arange(1, 17), metrics_kwiat_basis_lin_comb_lstm[metric_to_plot], label='LSTM with linear combination of Kwiat basis and loss')
    plt.plot(np.arange(1, 17), metrics_discrete_kwiat_basis_lstm[metric_to_plot], label='LSTM from discrete Kwiat basis')
    plt.plot(np.arange(1, 17), metrics_discrete_noise_break_kwiat_basis_lstm[metric_to_plot], label='LSTM from discrete Kwiat basis, noise turned off after 10 epochs')
    plt.plot(np.arange(1, 17), metrics_discrete_noise_break_unique_kwiat_basis_lstm[metric_to_plot], label='LSTM from discrete unique Kwiat basis, noise turned off after 10 epochs')
    # plt.plot(np.arange(1, 17), tomography_fixed_metrics, label='Kwiat basis tomography')
    # plt.plot(np.arange(1, 17), zeroed_tomography_fixed_metrics, label='Kwiat basis tomography with zeroed measurements')
    # plt.plot(np.arange(1, 17), mle_intensity_fixed_metrics, label='Kwiat basis MLE with intensity')
    # plt.plot(np.arange(1, 17), mle_fixed_metrics, label='Kwiat basis MLE')
    # plt.plot(np.arange(1, 17), metrics_basis_gammas['test_loss_avg'], label='Tomography with measurement projector gammas')
    # plt.plot(np.arange(1, 17), metrics_pinv_gammas['test_loss_avg'], label='Tomography with pseudoinverse')
    plt.plot(np.arange(1, 17), metrics_tomography_corrections['test_loss_avg'], label='Tomography corrections predictor')


    plt.xticks(np.arange(1, 17))
    plt.title('MSE for reconstructed density matrix')
    plt.xlabel('Number of measurements')
    plt.ylabel('MSE') 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight')

if __name__ == '__main__':
    main()
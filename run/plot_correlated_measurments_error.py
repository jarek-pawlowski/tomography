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
    log_path_tomography = './logs/rho_varying_multiple_measurements/rho_test_varying_measurement_clipped.log'
    log_path_mle = './logs/rho_varying_multiple_measurements/rho_test_varying_measurement_clipped_optimized_intensity.log'
    plot_path = './plots/correlated_measurements_error.png'

    # load data
    metrics_lstm = load_metrics_from_file(log_path_lstm)
    metrics_smp = load_metrics_from_file(log_path_smp)
    metrics_tomography = load_metrics_from_file(log_path_tomography)
    metrics_mle = load_metrics_from_file(log_path_mle)

    # add metric for all correct measurements in tomography
    tomography_fixed_metrics = np.insert(metrics_tomography['bures_distance'], 0, 0)
    tomography_fixed_metrics = np.flip(tomography_fixed_metrics)[1:]

    mle_fixed_metrics = np.insert(metrics_mle['bures_distance'], 0, 0)
    mle_fixed_metrics = np.flip(mle_fixed_metrics)[1:]

    # plot
    plt.plot(np.arange(1, 17), metrics_lstm['bures_distance'], label='LSTM')
    plt.plot(np.arange(1, 17), metrics_smp['bures_distance'], label='Fully connected NN')
    plt.plot(np.arange(1, 17), tomography_fixed_metrics, label='Tomography')
    plt.plot(np.arange(1, 17), mle_fixed_metrics, label='MLE')

    plt.xticks(np.arange(1, 17))
    plt.title('Bures distance for reconstructed density matrix')
    plt.xlabel('Number of measurements')
    plt.ylabel('Bures distance') 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(plot_path, bbox_inches='tight')

if __name__ == '__main__':
    main()
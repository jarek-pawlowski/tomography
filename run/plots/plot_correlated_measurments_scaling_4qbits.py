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

    plt.rcParams.update({'font.size': 12})
    # fig, ax = plt.subplots()
    # ax.set_prop_cycle(color=[cm(10.*i/num_colors) for i in range(num_colors)])
    # plot
    plt.plot(metrics_pinv_gammas[xaxis_name], metrics_pinv_gammas[pinv_metrics_name], color='b', marker='h', markersize=6, fillstyle='none', linestyle='-.', label='Tomography with\npseudoinverse')
    plt.plot(metrics_corrections_predictor[xaxis_name], metrics_corrections_predictor[new_metric_name], color='orange', marker='o', linestyle='-', markersize=6, fillstyle='none', label='Corrector NN')

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
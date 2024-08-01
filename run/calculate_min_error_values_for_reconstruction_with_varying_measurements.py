import sys
sys.path.append('./')
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


from src.datasets import MeasurementDataset
from src.model import SequentialMeasurementPredictor, LSTMMeasurementPredictor
from src.torch_utils import train_measurement_predictor, test_measurement_predictor, torch_bures_distance
from src.logging import load_metrics_from_file, log_metrics_to_file



def load_metrics_with_optional_metrics_names(log_path: str, metrics_names: list):
    try:
        return load_metrics_from_file(log_path)
    except:
        return load_metrics_from_file(log_path, metrics_names)
    

def main():
    # set paths 
    dir_path = './logs/1qbit/rho_varying_zeroed_multiple_measurements/'
    save_path = './logs/1qbit/rho_test_varying_zeroed_measurement_clipped_tomography_min.log'

    filenames = os.listdir(dir_path)
    metrics_names = ['num measurements', 'test_rmse_loss', 'test_mse_loss', 'bures_distance']
    metrics = [load_metrics_with_optional_metrics_names(os.path.join(dir_path, filename), metrics_names) for filename in filenames]
    metrics_grouped_by_num_measurements = defaultdict(list)
    for metric in metrics:
        metrics_grouped_by_num_measurements[metric['num measurements'][0]].append(metric) 

    min_metrics_for_num_measurements = {int(num_measurements): {metric_name: min([metric[metric_name][0] for metric in metrics]) for metric_name in metrics_names[1:]} for num_measurements, metrics in metrics_grouped_by_num_measurements.items()}
    sorted_min_metrics_for_num_measurements = sorted(min_metrics_for_num_measurements.items(), key=lambda x: x[0])
    write_mode = 'w'
    for num_measurements, metrics in sorted_min_metrics_for_num_measurements:
        log_metrics_to_file(metrics, save_path, write_mode, xaxis=num_measurements, xaxis_name='num measurements')
        write_mode = 'a'


if __name__ == '__main__':
    main()
import sys
sys.path.append('./')
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from src.logging import load_metrics_from_file, log_metrics_to_file



def load_metrics_with_optional_metrics_names(log_path: str, metrics_names: list):
    try:
        return load_metrics_from_file(log_path)
    except:
        return load_metrics_from_file(log_path, metrics_names)
    

def main():
    # set paths 
    dir_prefix_path = './logs/1qbit/tomography_corrections_predictor_m'
    save_path = './logs/1qbit/tomography_corrections_predictor_subset_min.log'
    metrics_names = ['train_loss', 'test_loss', 'bures_distance']

    write_mode = 'w'
    for num_measurements in range(1, 5):
        dir_path = f'{dir_prefix_path}{num_measurements}/'
        filenames = os.listdir(dir_path)
        metrics = [load_metrics_from_file(os.path.join(dir_path, filename)) for filename in filenames]
        min_metrics = {metric_name: min([metric[metric_name][4] for metric in metrics]) for metric_name in metrics_names}
        log_metrics_to_file(min_metrics, save_path, write_mode, xaxis=num_measurements, xaxis_name='num measurements')
        write_mode = 'a'


if __name__ == '__main__':
    main()
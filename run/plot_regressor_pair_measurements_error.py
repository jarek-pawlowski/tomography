import sys
sys.path.append('./')

from src.logging import plot_metrics_from_file, plot_metrics_from_files

results_path_prefix = './logs/regressor_varying_pair_measurements/regressor_test_varying_measurement_clipped_'
results_path = '{}{}.log'
plot_path = './plots/regressor_varying_pair_measurements/regressor_test_varying_measurement_clipped_{}.png'

measurments_num = 16
predefined_measurements = [0, 4, 6, 15]


for k in predefined_measurements:
    for i in range(0, measurments_num):
        plot_metrics_from_file(results_path.format(results_path_prefix,  f'{k}-{i}'), title=f'Metrics for measurement {k}-{i}', save_path=plot_path.format(f'{k}-{i}'), xaxis='variance')

    plot_metrics_from_files(f'{results_path_prefix}{k}-', (0, measurments_num), title=f'RMSE loss for varying measurements {k}', save_path=plot_path.format(f'{k}-{i}_rmse'), xaxis='variance', specified_metric='test_rmse_loss')
    plot_metrics_from_files(f'{results_path_prefix}{k}-', (0, measurments_num), title=f'Accuracy for varying measurements {k}', save_path=plot_path.format(f'{k}-{i}_acc'), xaxis='variance', specified_metric='test_accuracy')

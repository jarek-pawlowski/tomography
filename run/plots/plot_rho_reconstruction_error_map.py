import numpy as np

from src.logging import plot_grouped_error_map, load_metrics_from_file


nn_results_path = './logs/1qbit/tomography_corrections_predictor_m2/mlp_tomography_corrections_predictor_m{}_complex_distance.log'
tomo_results_path = './logs/1qbit/density_matrix_reconstructor_from_pinv_gammas_v2/complex_distance_reconstruction_from_m{}.log'

nn_plot_path = './plots/final_results/1qbit_nn_tomo_corrections_complex_distance.png'
tomo_plot_path = './plots/final_results/1qbit_tomo_pinv_complex_distance.png'

num_measurements = 4
epoch_num = 5

measurement_subsets_str = ['0_1', '0_2', '0_3', '1_2', '1_3', '2_3']
metrics_names = ['00', '01', '10', '11']
nn_metrics_values = np.nan * np.ones(((num_measurements - 1) ** 2, len(metrics_names)))
tomo_metrics_values = np.nan * np.ones(((num_measurements - 1) ** 2, len(metrics_names)))

for i in range(num_measurements):
    for j in range(i + 1, num_measurements):
        measurement_subset_str = f'{i}_{j}'
        nn_metrics = load_metrics_from_file(nn_results_path.format(measurement_subset_str))
        nn_metrics_values[i * (num_measurements - 1) + j - 1, :] = [nn_metrics[metric_name][epoch_num - 1] for metric_name in metrics_names]
        tomo_metrics = load_metrics_from_file(tomo_results_path.format(measurement_subset_str))
        tomo_metrics_values[i * (num_measurements - 1) + j - 1, :] = [tomo_metrics[metric_name][0] for metric_name in metrics_names]

plot_grouped_error_map(nn_metrics_values, title='NN corrections complex distance', save_path=nn_plot_path, values_range=(0, 0.7))
plot_grouped_error_map(tomo_metrics_values, title='Tomography pinv complex distance', save_path=tomo_plot_path, values_range=(0, 0.7))

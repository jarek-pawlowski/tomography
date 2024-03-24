from src.logging import multiplot_metrics_from_files, plot_metrics_from_files, plot_map_from_files


regressor_results_path_prefix = './logs/vector_density_matrix_regressor_varying_rho_elements/regressor_test_varying_rho_clipped_'
tomography_results_path_prefix = './logs/vector_density_matrix_regressor_varying_rho_elements/concurrence_test_varying_measurement_clipped_'

regressor_plot_path = './plots/concurrence_vector_density_matrix_regressor_error_varying_rho{}.png'
tomography_plot_path = './plots/concurrence_tomography_error_varying_measurement{}.png'
multiplot_path = './plots/concurrence_error_varying_measurement{}.png'
num_measurements = 16


# multiplot_metrics_from_files(
#     path_prefixes=[regressor_results_path_prefix, tomography_results_path_prefix],
#     ranges=[(0, num_measurements), (0, num_measurements)],
#     label_prefixes=['Regressor ', 'Tomography '],
#     title=f'RMSE loss for concurrence reconstruction',
#     save_path=multiplot_path.format(f'_rmse'),
#     xaxis='variance',
#     specified_metric='test_rmse_loss',
#     linestyles=['solid', 'dashed']
# )
# multiplot_metrics_from_files(
#     path_prefixes=[regressor_results_path_prefix, tomography_results_path_prefix],
#     ranges=[(0, num_measurements), (0, num_measurements)],
#     label_prefixes=['Regressor ', 'Tomography '],
#     title=f'Accuracy for concurrence reconstruction',
#     save_path=multiplot_path.format(f'_acc'),
#     xaxis='variance',
#     specified_metric='test_accuracy',
#     linestyles=['solid', 'dashed']
# )
# plot_metrics_from_files(regressor_results_path_prefix, (0, num_measurements), title=f'RMSE loss for concurrence reconstruction with regressor', save_path=regressor_plot_path.format(f'_rmse'), xaxis='variance', specified_metric='test_rmse_loss')
# plot_metrics_from_files(regressor_results_path_prefix, (0, num_measurements), title=f'Accuracy for concurrence reconstruction with regressor', save_path=regressor_plot_path.format(f'_acc'), xaxis='variance', specified_metric='test_accuracy')
# plot_metrics_from_files(tomography_results_path_prefix, (0, num_measurements), title=f'RMSE loss for concurrence reconstruction with tomography', save_path=tomography_plot_path.format(f'_rmse'), xaxis='variance', specified_metric='test_rmse_loss')
# plot_metrics_from_files(tomography_results_path_prefix, (0, num_measurements), title=f'Accuracy for concurrence reconstruction with tomography', save_path=tomography_plot_path.format(f'_acc'), xaxis='variance', specified_metric='test_accuracy')

plot_map_from_files(
    regressor_results_path_prefix,
    xaxis='variance',
    xvalue=0.5,
    metric_name='test_rmse_loss',
    range_=(0, num_measurements),
    save_path=regressor_plot_path.format(f'_rmse_map'),
    values_range=(0.05, 0.23)
)

# plot_map_from_files(
#     tomography_results_path_prefix,
#     xaxis='variance',
#     xvalue=0.5,
#     metric_name='test_rmse_loss',
#     range_=(0, num_measurements),
#     save_path=tomography_plot_path.format(f'_rmse_map'),
#     values_range=(0.095, 0.113)
# )
from src.logging import multiplot_metrics_from_files, plot_metrics_from_files, plot_map_from_files, plot_map_from_file


regressor_results_path_prefix = './logs/regressor_varying_measurements/regressor_test_varying_measurement_clipped_'
tomography_results_path_prefix = './logs/concurrence_varying_measurements/concurrence_test_varying_measurement_clipped_'

regressor_feature_importance_path = 'logs/regressor_features_importance/real_distribution_range_features_importance_with_varied_input_mean_distance_dataset_values.log'
tomography_feature_importance_path = 'logs/tomography_features_importance/real_distribution_range_features_importance_with_varied_input_mean_distance_dataset_values.log'

regressor_plot_path = './plots/final_results/concurrence_regressor_error_varying_measurement{}.png'
tomography_plot_path = './plots/final_results/concurrence_tomography_error_varying_measurement{}.png'
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
    # values_range=(0.05, 0.23),
    style='seaborn',
    # title='RMSE loss for concurrence reconstruction with regressor'
)

plot_map_from_files(
    tomography_results_path_prefix,
    xaxis='variance',
    xvalue=0.5,
    metric_name='test_rmse_loss',
    range_=(0, num_measurements),
    save_path=tomography_plot_path.format(f'_rmse_map'),
    # values_range=(0.095, 0.113),
    style='seaborn',
    # title='RMSE loss for concurrence reconstruction with tomography'
)

plot_map_from_file(
    regressor_feature_importance_path,
    metric_name='mean',
    save_path=regressor_plot_path.format(f'_feature_importance_map'),
    # values_range=(1.11, 1.33),
    # title='Feature importance for concurrence reconstruction with regressor',
    style='seaborn'
)

plot_map_from_file(
    tomography_feature_importance_path,
    metric_name='mean',
    save_path=tomography_plot_path.format(f'_feature_importance_map'),
    # values_range=(0.06, 0.073),
    # title='Feature importance for concurrence reconstruction with regressor',
    style='seaborn'
)
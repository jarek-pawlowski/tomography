import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.log import multiplot_metrics_from_files, plot_metrics_from_files, plot_map_from_files, plot_map_from_file


regressor_results_path_prefix = './logs/regressor_varying_measurements/regressor_test_varying_measurement_clipped_'

regressor_results_filtered_path_prefix = './logs/regressor_varying_measurements/regressor_test_varying_measurements_clipped_train_filtered'
# tomography_results_path_prefix = './logs/concurrence_varying_measurements/concurrence_test_varying_measurement_clipped_filtered_data_'

regressor_feature_importance_path = 'logs/regressor_features_importance/real_distribution_noise_features_importance_with_varied_feature_input_mean_mad.log'
tomography_feature_importance_path = 'logs/tomography_features_importance/real_distribution_noise_features_importance_with_varied_feature_input_mean_mad.log'


regressor_plot_path = './plots/final_results/E_p_tomo_NN_v3.png'
multiplot_path = './plots/concurrence_error_varying_measurement_with_noise{}.png'
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


############################################################################################
# Regressor plot
############################################################################################

fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

plot_map_from_files(
    regressor_results_path_prefix,
    xaxis='variance',
    xvalue=0.5,
    metric_name='test_rmse_loss',
    range_=(0, num_measurements),
    # save_path=regressor_plot_path.format(f'_rmse_map'),
    values_range=(0.125, 0.24),
    style='seaborn',
    # title='RMSE loss for concurrence reconstruction with regressor'
    global_ax=ax1,
    close=False
)

plot_map_from_files(
    regressor_results_filtered_path_prefix,
    xaxis='variance',
    xvalue=0.5,
    metric_name='test_rmse_loss',
    range_=(0, num_measurements),
    # save_path=tomography_plot_path.format(f'_rmse_map'),
    values_range=(0.015, 0.069),
    style='seaborn',
    # title='RMSE loss for concurrence reconstruction with tomography',
    global_ax=ax2,
    close=False
)

ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax1.tick_params(axis='y', labelrotation=1.)
ax2.tick_params(axis='y', labelrotation=1.)

# Create a colorbar axis above the plots

# Add colorbar with label
# cbar = plt.colorbar(ax1.collections[0], cax=cbar_ax, orientation="horizontal")
# cbar.set_label('Feature importance', fontsize=20)
# cbar.ax.xaxis.set_label_coords(0.3, -1.4)

ax1.text(0.25, 1.07, "(c)  $E^{P}_{NN}$($\\bf{S}_{test}$)", transform=ax1.transAxes, fontsize=20, va='center', ha='center')
ax2.text(0.27, 1.07, "(d)  $E^{P}_{NN}$($\\bf{S'}_{test}$)", transform=ax2.transAxes, fontsize=20, va='center', ha='center')

plt.savefig(regressor_plot_path, format="png", bbox_inches="tight", dpi=500)
plt.close()

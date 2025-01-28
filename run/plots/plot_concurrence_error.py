import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.logging import multiplot_metrics_from_files, plot_metrics_from_files, plot_map_from_files, plot_map_from_file


regressor_results_path_prefix = './logs/regressor_varying_measurements/regressor_test_varying_measurement_clipped_'
tomography_results_path_prefix = './logs/concurrence_varying_measurements/concurrence_test_varying_measurement_clipped_'

regressor_feature_importance_path = 'logs/regressor_features_importance/real_distribution_noise_features_importance_with_varied_feature_input_mean_mad.log'
tomography_feature_importance_path = 'logs/tomography_features_importance/real_distribution_noise_features_importance_with_varied_feature_input_mean_mad.log'


regressor_plot_path = './plots/final_results/concurrence_regressor_error_varying_measurement_mean_mad{}.png'
tomography_plot_path = './plots/final_results/concurrence_tomography_error_varying_measurement_mean_mad{}.png'
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

plot_map_from_file(
    regressor_feature_importance_path,
    metric_name='mean_mad',
    # save_path=regressor_plot_path.format(f'_feature_importance_map'),
    values_range=(0.093, 0.174),
    # title='Feature importance for concurrence reconstruction with regressor',
    style='seaborn',
    global_ax=ax1,
    close=False
)

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

# Add colorbar for ax1 over the plot, adjust the position of the colorbar
cbar_ax1 = fig.add_axes([0.14, 0.9, 0.34, 0.05]) 
cbar = plt.colorbar(ax1.collections[0], cax=cbar_ax1, orientation="horizontal")
# reduce the size of the colorbar
cbar.ax.tick_params(labelsize=18)
# move the colorbar to the top of the plot
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_coords(0.5, 3)
cbar.set_label('$E^{S}$', fontsize=20)
# shrink the colorbar to fit the plot

cbar_ax2 = fig.add_axes([0.54, 0.9, 0.34, 0.05]) 
cbar = plt.colorbar(ax2.collections[0], cax=cbar_ax2, orientation="horizontal")
# reduce the size of the colorbar
cbar.ax.tick_params(labelsize=18)
# move the colorbar to the top of the plot
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_coords(0.5, 3)
cbar.set_label('$E^{P}$', fontsize=20)

ax1.text(0.1, 1.23, '(a)', transform=ax1.transAxes, fontsize=20, va='center', ha='center')
ax2.text(0.1, 1.23, '(b)', transform=ax2.transAxes, fontsize=20, va='center', ha='center')

plt.savefig(regressor_plot_path.format(f'_maps'), format="png", bbox_inches="tight")
plt.close()


############################################################################################
# Tomography plot
############################################################################################

fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

plot_map_from_file(
    tomography_feature_importance_path,
    metric_name='mean_mad',
    # save_path=tomography_plot_path.format(f'_feature_importance_map'),
    values_range=(0.0412, 0.0451),
    # title='Feature importance for concurrence reconstruction with regressor',
    style='seaborn',
    global_ax=ax1,
    close=False
)

plot_map_from_files(
    tomography_results_path_prefix,
    xaxis='variance',
    xvalue=0.5,
    metric_name='test_rmse_loss',
    range_=(0, num_measurements),
    # save_path=tomography_plot_path.format(f'_rmse_map'),
    values_range=(0.098, 0.112),
    style='seaborn',
    # title='RMSE loss for concurrence reconstruction with tomography',
    global_ax=ax2,
    close=False
)

# ax1.tick_params(axis='both', which='major', labelsize=18)
# ax2.tick_params(axis='both', which='major', labelsize=18)
# ax1.tick_params(axis='y', labelrotation=1.)
# ax2.tick_params(axis='y', labelrotation=1.)

# # Create a colorbar axis above the plots
# cbar_ax = fig.add_axes([0.2, 1.06, 0.6, 0.05]) 
# cbar_ax.tick_params(axis='both', which='major', labelsize=18)

# # Add colorbar with label
# cbar = plt.colorbar(ax1.collections[0], cax=cbar_ax, orientation="horizontal")
# cbar.set_label('Feature importance', fontsize=20)
# cbar.ax.xaxis.set_label_coords(0.3, -1.4)

cbar_ax1 = fig.add_axes([0.14, 0.9, 0.34, 0.05]) 
cbar = plt.colorbar(ax1.collections[0], cax=cbar_ax1, orientation="horizontal")
# reduce the size of the colorbar
cbar.ax.tick_params(labelsize=18)
# move the colorbar to the top of the plot
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_coords(0.5, 3)
cbar.set_label('$E^{S}$', fontsize=20)
# shrink the colorbar to fit the plot

cbar_ax2 = fig.add_axes([0.54, 0.9, 0.34, 0.05]) 
cbar = plt.colorbar(ax2.collections[0], cax=cbar_ax2, orientation="horizontal")
# reduce the size of the colorbar
cbar.ax.tick_params(labelsize=18)
# move the colorbar to the top of the plot
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_coords(0.5, 3)
cbar.set_label('$E^{P}$', fontsize=20)

ax1.text(0.1, 1.23, '(a)', transform=ax1.transAxes, fontsize=20, va='center', ha='center')
ax2.text(0.1, 1.23, '(b)', transform=ax2.transAxes, fontsize=20, va='center', ha='center')

plt.savefig(tomography_plot_path.format(f'_maps'), format="png", bbox_inches="tight")
plt.close()
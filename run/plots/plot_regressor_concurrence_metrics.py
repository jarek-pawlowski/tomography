import os
import sys

import numpy as np
sys.path.append('./')

import matplotlib.pyplot as plt

from src.logging import plot_metrics_from_file, plot_metrics_from_files, load_metrics_from_file

results_nn_path = './logs/regressor_test_conc.log'
results_rf_path = './logs/rf1_accuracy.csv'
plot_path = './plots/con_dependence/final_acc.png'

plt.rcParams.update({'font.size': 12})

metrics_names = ['test_accuracy']
# plot_metrics_from_file(results_nn_path, title=f'Metrics for measurement', save_path=plot_path, xaxis='concurrence', metrics_names=metrics_names)

plt.figure(figsize=(6, 4))

metrics_nn = load_metrics_from_file(results_nn_path)
nn_conc = np.concatenate((metrics_nn["concurrence"][:17], metrics_nn["concurrence"][17:][::5]))
nn_acc = np.concatenate((metrics_nn["test_accuracy"][:17], metrics_nn["test_accuracy"][17:][::5]))
plt.plot(nn_conc, nn_acc, label='NN', color='b', marker='o', markersize=5, fillstyle='none', linestyle='--', linewidth=1.5)

metrics_rf = load_metrics_from_file(results_rf_path, delimiter=',')
rf_conc = np.concatenate((metrics_rf["C"][:28], metrics_rf["C"][28:][::5]))
rf_acc = np.concatenate((metrics_rf["acc_pos"][:28], metrics_rf["acc_pos"][28:][::5]))
plt.plot(rf_conc, rf_acc, label='RF', color='orange', marker='x', linestyle=(0, (3, 3)), markersize=6, fillstyle='none', linewidth=1.5)

plt.ylim(0.6)
plt.xlabel("Concurrence")
plt.ylabel("ACC ($\\bf{S}_{test}$)")
plt.legend()
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=500, bbox_inches='tight')

import os
import pickle

from matplotlib import gridspec, pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import MeasurementDataset
from src.logging import plot_error_map_seaborn
from src.tomography_utils_torch import calculate_concurrence_from_measurements


metrics_pickle_path = './shapley/tomo_lib_shap_values_kernel_exp_500.pkl'


try:
    with open(metrics_pickle_path, 'rb') as f:
        metrics = pickle.load(f)
        print(f"Metrics loaded from {metrics_pickle_path}")
except FileNotFoundError:
    train_batch_size = 1000
    train_dataset = MeasurementDataset(root_path='./data/train/')
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    
    test_batch_size = 40000
    test_dataset = MeasurementDataset(root_path='./data/val/')
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)


    threshold_sep = 1.e-6
    threshold_ent = 0.99

    train_tensors, _ = next(iter(train_loader))
    train_sample = shap.kmeans(train_tensors.numpy(), 100)
    explainer = shap.KernelExplainer(calculate_concurrence_from_measurements, train_sample)

    data_bin_names = ['all', 'sep', 'ent', 'both']
    criterions = {
        'mean': lambda x: np.mean(x, axis=0),
        'std': lambda x: np.std(x, axis=0),
        'median': lambda x: np.median(x, axis=0),
    }

    test_tensors, test_labels = next(iter(test_loader))

    shap_values_all = explainer.shap_values(test_tensors[:1000].numpy())
        
    mask_sep = test_labels.squeeze() < threshold_sep
    shap_values_sep = explainer.shap_values(test_tensors[mask_sep][:1000].numpy())

    mask_ent = test_labels.squeeze() > threshold_ent
    shap_values_ent = explainer.shap_values(test_tensors[mask_ent][:1000].numpy())

    # mask_both = mask_sep | mask_ent
    # both_tensors = torch.cat((test_tensors[mask_sep][:5000], test_tensors[mask_ent][:5000]))
    # shap_values_both = explainer.shap_values(both_tensors.numpy())

    metrics = {}
    for key, criterion in criterions.items():
        metrics[f'all_{key}'] = criterion(shap_values_all)
        metrics[f'sep_{key}'] = criterion(shap_values_sep)
        metrics[f'ent_{key}'] = criterion(shap_values_ent)
        # metrics[f'both_{key}'] = criterion(shap_values_both)

    os.makedirs(os.path.dirname(metrics_pickle_path), exist_ok=True)
    with open(metrics_pickle_path, 'wb') as f:
        pickle.dump(metrics, f)


fig = plt.figure(figsize=(15, 7))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

sep_values = metrics['sep_mean']
plot_error_map_seaborn(sep_values, close=False, global_ax=ax1)

ent_values = metrics['ent_mean']
plot_error_map_seaborn(ent_values, close=False, global_ax=ax2)

ax1.tick_params(axis='both', which='major', labelsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax1.tick_params(axis='y', labelrotation=1.)
ax2.tick_params(axis='y', labelrotation=1.)

ax1.text(0.3, 1.07, '(e)  $\Phi_{NN}$ for C < $10^{-6}$', transform=ax1.transAxes, fontsize=20, va='center', ha='center')
ax2.text(0.3, 1.07, '(f)  $\Phi_{NN}$ for C > 0.99', transform=ax2.transAxes, fontsize=20, va='center', ha='center')

plt.savefig('./plots/shapley_values_tomo_kernel_exp/NN_labels.png', format="png", bbox_inches="tight", dpi=500)
plt.close()

# # Plot results
# for metric_name, metric_values in metrics.items():
#     abs_metrics_values = np.abs(metric_values)
#     plot_error_map_seaborn(metric_values, save_path=f'./plots/shapley_values_tomo_kernel_exp/{metric_name}_500_replot.png', title="Shapley values")
#     plot_error_map_seaborn(abs_metrics_values, save_path=f'./plots/shapley_values_tomo_kernel_exp/abs_{metric_name}_500_replot.png', title="Shapley values")

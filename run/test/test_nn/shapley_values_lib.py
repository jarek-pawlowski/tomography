import os
import pickle
from matplotlib import gridspec, pyplot as plt
import numpy as np
import shap
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import MeasurementDataset
from src.logging import plot_error_map_seaborn
from src.model import Regressor


metrics_pickle_path = './shapley/regressor_lib_shap_values_10000.pkl'

try:
    with open(metrics_pickle_path, 'rb') as f:
        metrics = pickle.load(f)
        print(f"Metrics loaded from {metrics_pickle_path}")
except FileNotFoundError:
    train_batch_size = 10000
    train_dataset = MeasurementDataset(root_path='./data/train/')
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    
    test_batch_size = 512
    test_dataset = MeasurementDataset(root_path='./data/val/')
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    model_path = './models/regressor.pt'
    model_params = {
        'input_dim': 16,
        'output_dim': 1,
        'layers': 2,
        'hidden_size': 128,
        'input_dropout': 0.
    }
    model = Regressor(**model_params)
    model.load(model_path)


    threshold_sep = 1.e-6
    threshold_ent = 0.99

    train_tensors, train_labels = next(iter(train_loader))
    shap_explainer = shap.DeepExplainer(model, train_tensors)

    data_bin_names = ['all', 'sep', 'ent', 'both']
    criterions = {
        'mean': lambda x: np.mean(x, axis=0),
        'std': lambda x: np.std(x, axis=0),
        'median': lambda x: np.median(x, axis=0),
    }
    metrics = {
        f'{bin_name}_{criterion_name}': [] for bin_name in data_bin_names for criterion_name in criterions.keys()
    }

    # Arbitrary samples
    for data, labels in tqdm(test_loader, desc='Calculating Shapley values'):
        shap_values_all = shap_explainer.shap_values(data)
            
        mask_sep = labels.squeeze() < threshold_sep
        shap_values_sep = shap_explainer.shap_values(data[mask_sep]).squeeze()

        mask_ent = labels.squeeze() > threshold_ent
        shap_values_ent = shap_explainer.shap_values(data[mask_ent]).squeeze()

        mask_both = mask_sep | mask_ent
        shap_values_both = shap_explainer.shap_values(data[mask_both]).squeeze()

        for key, criterion in criterions.items():
            metrics[f'all_{key}'].append(criterion(shap_values_all))
            metrics[f'sep_{key}'].append(criterion(shap_values_sep))
            metrics[f'ent_{key}'].append(criterion(shap_values_ent))
            metrics[f'both_{key}'].append(criterion(shap_values_both))

    # Aggregate results
    for key, values in metrics.items():
        metrics[key] = np.mean(values, axis=0)

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

ax1.text(0.3, 1.07, '(e)  $\Phi_{i}$ for C < $10^{-6}$', transform=ax1.transAxes, fontsize=20, va='center', ha='center')
ax2.text(0.3, 1.07, '(f)  $\Phi_{i}$ for C > 0.99', transform=ax2.transAxes, fontsize=20, va='center', ha='center')

plt.savefig('./plots/shapley_values_abs/sep_and_ent_10000.png', format="png", bbox_inches="tight", dpi=500)
plt.close()

# Plot results
for metric_name, metric_values in metrics.items():
    abs_metrics_values = np.abs(metric_values)
    plot_error_map_seaborn(abs_metrics_values, save_path=f'./plots/shapley_values_abs/{metric_name}_10000.png', title="Shapley values")
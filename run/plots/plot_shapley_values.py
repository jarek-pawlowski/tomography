import os
import pickle
import typing as t

from matplotlib import gridspec, pyplot as plt
import numpy as np
import seaborn as sns


def plot_error_map_seaborn(
    metrics_values: np.ndarray,
    save_path: t.Optional[str] = None,
    values_range: t.Optional[t.Tuple[float, float]] = None,
    title: str = '',
    close: bool = True,
    global_ax = None,
):
    params = {
        "annot": metrics_values.reshape((4,4)), 
        "fmt": ".4f", 
        "linewidth": 3.0, 
        "annot_kws": {"size":20},
        "cbar": False,
        "square": True
    }
    if values_range is not None:
        params['vmin'] = values_range[0]
        params['vmax'] = values_range[1]
    if global_ax is None:
        plt.figure(figsize=(15,12))
    sns.set_theme(font_scale=1.7)
    sns.heatmap(
        abs(metrics_values.reshape((4,4))),
        **params,
        ax=global_ax
    )
    plt.title(title, size=22)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="png", bbox_inches="tight")
    plt.show()
    if close:
        plt.close()


# metrics_pickle_path = './shapley/tomo_lib_shap_values_kernel_exp_500.pkl'
metrics_pickle_path = './shapley/regressor_lib_shap_values_10000.pkl'


with open(metrics_pickle_path, 'rb') as f:
    metrics = pickle.load(f)

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

plt.savefig('./plots/shapley_values_final/shapley_NN.png', format="png", bbox_inches="tight", dpi=500)
plt.close()
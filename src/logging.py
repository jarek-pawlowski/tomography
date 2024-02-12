import os
import typing as t
from math import floor, sqrt

import matplotlib.pyplot as plt
import numpy as np

DELIMITER = ', '

def log_metrics_to_file(metrics: t.Dict[str, float], log_path: str, write_mode: str = 'w', xaxis: t.Optional[float] = None, xaxis_name: str = 'epoch') -> None:    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, write_mode) as f:
        if write_mode == 'w':
            prefix = f'{xaxis_name}, ' if xaxis is not None else ''
            f.write(f'{prefix}{str.join(DELIMITER, metrics.keys())}\n')

        prefix = f'{xaxis}, ' if xaxis is not None else ''
        metrics_values_str = map(lambda x: str(x), metrics.values())
        f.write(f'{prefix}{str.join(DELIMITER, metrics_values_str)}\n')


def load_metrics_from_file(log_path: str) -> t.Dict[str, np.ndarray]:
    with open(log_path, 'r') as f:
        data = f.readlines()
    data = [x.strip().split(DELIMITER) for x in data]
    metrics_names = data[0]
    data = np.array([[float(y) for y in x] for x in data[1:]])
    return {name: data[:, i] for i, name in enumerate(metrics_names)}


def plot_metrics_from_file(log_path: str, title: str = '', save_path: t.Optional[str] = None, xaxis: str = 'epoch') -> None:
    metrics = load_metrics_from_file(log_path)
    epochs = metrics.pop(xaxis)
    for metric_name, metric_value in metrics.items():
        plt.plot(epochs, metric_value, label=metric_name)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.legend()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def plot_metrics_from_files(path_common_prefix: str, range_: t.Tuple[int, int], title: str = '', save_path: t.Optional[str] = None, xaxis: str = 'epoch', specified_metric: t.Optional[str] = None) -> None:
    plot_metrics_from_common_prefix(path_common_prefix, range_, xaxis, specified_metric)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_metrics_from_common_prefix(path_common_prefix: str, range_: t.Tuple[int, int], xaxis: str = 'epoch', specified_metric: t.Optional[str] = None, label_prefix: str = '', linestyle='solid') -> None:
    for i in range(*range_):
        log_path = f'{path_common_prefix}{i}.log'
        metrics = load_metrics_from_file(log_path)
        epochs = metrics.pop(xaxis)
        matrix_dim = int(sqrt(range_[1]))
        label_i = f'{floor(i / matrix_dim)}_{i % matrix_dim}'
        # set color according to i and color palette
        cmap = plt.get_cmap('tab20')
        color = cmap(i) 
        if specified_metric is not None:
            plt.plot(epochs, metrics[specified_metric], label=f'{label_prefix}{specified_metric} {label_i}', color=color, linestyle=linestyle)            
        else:
            for metric_name, metric_value in metrics.items():
                plt.plot(epochs, metric_value, label=f'{label_prefix}{metric_name} {label_i}', color=color, linestyle=linestyle)


def multiplot_metrics_from_files(
    path_prefixes: t.List[str],
    ranges: t.List[t.Tuple[int, int]],
    label_prefixes: t.List[str],
    title: str = '',
    save_path: t.Optional[str] = None,
    xaxis: str = 'epoch',
    specified_metric: t.Optional[str] = None,
    linestyles: t.Optional[t.List[str]] = None
) -> None:
    if linestyles is None:
        linestyles = ['solid'] * len(path_prefixes)
    for path_prefix, range_, label_prefix, linestyle in zip(path_prefixes, ranges, label_prefixes, linestyles):
        plot_metrics_from_common_prefix(path_prefix, range_, xaxis, specified_metric, label_prefix, linestyle)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_map_from_files(
    path_common_prefix: str,
    xaxis: str,
    xvalue: float,
    metric_name: str,
    range_: t.Tuple[int, int],
    save_path: t.Optional[str] = None,
    values_range: t.Optional[t.Tuple[float, float]] = None
):
    # plot matrix of metrics values for each file in range_ and value closest to xvalue
    matrix_dim = int(sqrt(range_[1]))
    fig, axes = plt.subplots(matrix_dim, matrix_dim, figsize=(10, 10))
    cmap = plt.get_cmap('viridis')
    if values_range is not None:
        norm = plt.Normalize(vmin=values_range[0], vmax=values_range[1])
    else:
        norm = plt.Normalize()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i in range(*range_):
        log_path = f'{path_common_prefix}{i}.log'
        metrics = load_metrics_from_file(log_path)
        metrics_values = metrics[metric_name]
        xvalues = metrics[xaxis]
        idx = (np.abs(xvalues - xvalue)).argmin()
        metric_value = metrics_values[idx]

        # plot metric value as a subplot square in matrix
        # use color palette to set color according to metric value
        # also plot the text value of the metric in the center of the square
        # set color map to pretty colors and scale color according to min and max of metric values
        color = cmap(norm(metric_value))

        label_i = f'{floor(i / matrix_dim)}_{i % matrix_dim}'

        axes.flat[i].text(0, 0, f'{metric_value:.4f}', ha='center', va='center', fontsize=8)
        axes.flat[i].set_xticks([0]) 
        axes.flat[i].set_xticklabels([i % matrix_dim])
        axes.flat[i].set_yticks([0]) 
        axes.flat[i].set_yticklabels([floor(i / matrix_dim)])
        axes.flat[i].grid(False)
        # axes.flat[i].xaxis.set_visible(False)
        # axes.flat[i].yaxis.set_visible(False)
        axes.flat[i].imshow([[color]])
        axes.flat[i].label_outer()

    # Set common xticks and yticks for subplots
    for ax in axes.flat:
        ax.label_outer()
        

    # show color bar on the global plot
    fig.subplots_adjust(wspace=0.1, hspace=0.01, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sm, cax=cbar_ax)
    plt.suptitle(f'{metric_name} for {xaxis}={xvalue}')
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

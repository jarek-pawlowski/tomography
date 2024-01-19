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
    plt.close()


def plot_metrics_from_files(path_common_prefix: str, range_: t.Tuple[int, int], title: str = '', save_path: t.Optional[str] = None, xaxis: str = 'epoch', specified_metric: t.Optional[str] = None) -> None:
    plot_metrics_from_common_prefix(path_common_prefix, range_, xaxis, specified_metric)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


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
    plt.close()

import os
import typing as t
from math import floor, sqrt

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def load_metrics_from_file(log_path: str, metrics_names: t.Optional[t.List[str]] = None, delimiter: str = DELIMITER) -> t.Dict[str, np.ndarray]:
    values_start_row = 0
    with open(log_path, 'r') as f:
        data = f.readlines()
    data = [x.strip().split(delimiter) for x in data]
    if metrics_names is None:
        metrics_names = data[0]
        values_start_row = 1
    values = np.array([[float(y) for y in x] for x in data[values_start_row:]])
    return {name: values[:, i] for i, name in enumerate(metrics_names)}


def plot_metrics_from_file(log_path: str, title: str = '', save_path: t.Optional[str] = None, xaxis: str = 'epoch', metrics_names: t.Optional[t.List[str]] = None, **kwargs: t.Dict[str, t.Any]) -> None:
    metrics = load_metrics_from_file(log_path)
    epochs = metrics.pop(xaxis)
    if metrics_names is None:
        metrics_names = metrics.keys()
    for metric_name in metrics_names:
        metric_value = metrics[metric_name]
        plt.plot(epochs, metric_value, label=metric_name, **kwargs)
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
    if xaxis == 'variance':
        plt.xlabel(r'$\sigma$')
    else:
        plt.xlabel(xaxis)
    if specified_metric is not None:
        if 'rmse' in specified_metric:
            plt.ylabel('RMSE')
        elif 'bce' in specified_metric:
            plt.ylabel('BCE')
        elif 'accuracy' in specified_metric:
            plt.ylabel('Accuracy')
        else:
            plt.ylabel(specified_metric)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.axvline(x=0.5, color='black', linestyle='--')
    plt.xticks(np.arange(0, 1.1, 0.1))
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
        index = f'{floor(i / matrix_dim)}{i % matrix_dim}'
        label_i = '$m_{' + index + '}$'
        # set color according to i and color palette
        cmap = plt.get_cmap('tab20')
        color = cmap(i) 
        if specified_metric is not None:
            plt.plot(epochs, metrics[specified_metric], label=f'{label_prefix}{label_i}', color=color, linestyle=linestyle)            
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


def plot_map_from_files(
    path_common_prefix: str,
    xaxis: str,
    xvalue: float,
    metric_name: str,
    range_: t.Tuple[int, int],
    save_path: t.Optional[str] = None,
    values_range: t.Optional[t.Tuple[float, float]] = None,
    style: str = 'seaborn', # 'seaborn' or 'matplotlib'
    title: str = '',
    global_ax = None,
    close: bool = True
):
    metrics_values_for_xvalue = []
    for i in range(*range_):
        log_path = f'{path_common_prefix}{i}.log'
        metrics = load_metrics_from_file(log_path)
        metrics_values = metrics[metric_name]
        xvalues = metrics[xaxis]
        idx = (np.abs(xvalues - xvalue)).argmin()
        metric_value = metrics_values[idx]
        metrics_values_for_xvalue.append(metric_value)
    metrics_values_for_xvalue = np.array(metrics_values_for_xvalue)
    if style == 'seaborn':
        plot_error_map_seaborn(metrics_values_for_xvalue, save_path, values_range, title=title, global_ax=global_ax, close=close)
    elif style == 'matplotlib':
        plot_error_map(metrics_values_for_xvalue, save_path, values_range, title=title, close=close)
    else:
        raise ValueError(f'Unrecognized style: {style}, should be one of: {["seaborn", "matplotlib"]}')


def plot_map_from_file(
    path_to_file: str,
    metric_name: str,
    save_path: t.Optional[str] = None,
    values_range: t.Optional[t.Tuple[float, float]] = None,
    title: str = '',
    style: str = 'seaborn', # 'seaborn' or 'matplotlib'
    global_ax = None,
    close: bool = True
):
    metrics = load_metrics_from_file(path_to_file)
    metrics_values = metrics[metric_name]
    if style == 'seaborn':
        plot_error_map_seaborn(metrics_values, save_path, values_range, title=title, global_ax=global_ax, close=close)
    elif style == 'matplotlib':
        plot_error_map(metrics_values, save_path, values_range, title=title, close=close)
    else:
        raise ValueError(f'Unrecognized style: {style}, should be one of: {["seaborn", "matplotlib"]}')


def plot_error_map(
    metrics_values: np.ndarray,
    save_path: t.Optional[str] = None,
    values_range: t.Optional[t.Tuple[float, float]] = None,
    title: str = '',
    close: bool = True
):

    # plot matrix of metrics values for each file in range_ and value closest to xvalue
    matrix_dim = int(sqrt(len(metrics_values)))
    fig, axes = plt.subplots(matrix_dim, matrix_dim, figsize=(10, 10))
    cmap = plt.get_cmap('viridis')
    if values_range is not None:
        norm = plt.Normalize(vmin=values_range[0], vmax=values_range[1])
    else:
        norm = plt.Normalize()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, metric_value in enumerate(metrics_values):
        # plot metric value as a subplot square in matrix
        # use color palette to set color according to metric value
        # also plot the text value of the metric in the center of the square
        # set color map to pretty colors and scale color according to min and max of metric values
        color = cmap(norm(metric_value))
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
        
    # show color bar on the global plot
    fig.subplots_adjust(wspace=0.1, hspace=0.01, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sm, cax=cbar_ax)
    plt.suptitle(title, fontsize=20)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    if close:
        plt.close()


def plot_grouped_error_map(
    metrics_values: np.ndarray,
    figsize: t.Tuple[int, int] = (10, 12),
    save_path: t.Optional[str] = None,
    values_range: t.Optional[t.Tuple[float, float]] = None,
    title: str = '',
    close: bool = True
):
    outer_dim, inner_dim = metrics_values.shape
    outer_dim_sqrt = int(sqrt(outer_dim))
    inner_dim_sqrt = int(sqrt(inner_dim))
    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(outer_dim_sqrt, outer_dim_sqrt, wspace=0.2, hspace=0.2)

    cmap = plt.get_cmap('YlGnBu')
    if values_range is not None:
        norm = plt.Normalize(vmin=values_range[0], vmax=values_range[1])
    else:
        norm = plt.Normalize()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.rcParams.update({'font.size': 20})

    for i in range(outer_dim):
        inner = gridspec.GridSpecFromSubplotSpec(inner_dim_sqrt, inner_dim_sqrt,
                        subplot_spec=outer[i], wspace=0., hspace=0.01)

        # set xticks and yticks for the outer plot
        outer_ax = plt.subplot(outer[i])
        outer_ax.set_xticks([0.5])
        outer_ax.set_xticklabels([i % outer_dim_sqrt + 1])
        outer_ax.set_yticks([0.5])
        outer_ax.set_yticklabels([floor(i / outer_dim_sqrt)])
        outer_ax.grid(False)
        outer_ax.label_outer()
        # fig.add_subplot(outer_ax)
        plt.box(False)
        
        for j in range(inner_dim):
            metric_value = metrics_values[i, j]
            if not np.isnan(metric_value):
                color = cmap(norm(metric_value))
                ax = plt.subplot(inner[j])
                ax.text(0, 0, f'{metric_value:.4f}', ha='center', va='center', fontsize=20)
                ax.imshow([[color]])
                ax.grid(False)
                if i== 0 and j == 0:
                    ax.set_yticks([0.5])
                    ax.set_yticklabels([floor(i / outer_dim_sqrt)])
                    ax.xaxis.set_visible(False)
                elif i == 8 and j == 2:
                    ax.set_xticks([0.5])
                    ax.set_xticklabels([i % outer_dim_sqrt + 1])
                    ax.yaxis.set_visible(False)
                else:
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                ax.label_outer()
                fig.add_subplot(ax)

    fig.subplots_adjust(wspace=0., hspace=0., top=0.93, bottom=0.28) #right=0.97)
    fig.supxlabel('m', y = 0.23, fontsize=20)
    fig.supylabel('m', x = 0.06, fontsize=20)

    cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_xlabel('$E_C$')
    plt.suptitle(title, fontsize=20)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=1000)
    plt.show()
    if close:
        plt.close()


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


def plot_matrices(
    matrices: np.ndarray,
    save_path: str,
    title: str = '',
    values_range: t.Optional[t.Tuple[float, float]] = None,
):
    # plot matrix of metrics values for each file in range_ and value closest to xvalue
    matrix_dim = int(sqrt(len(matrices)))
    fig, axes = plt.subplots(matrix_dim, matrix_dim, figsize=(10, 10))

    cmap = plt.get_cmap('viridis')
    if values_range is not None:
        norm = plt.Normalize(vmin=values_range[0], vmax=values_range[1])
    else:
        norm = plt.Normalize()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, matrix in enumerate(matrices):
        matrix_map = cmap(norm(matrix))

        axes.flat[i].imshow(matrix_map)
        axes.flat[i].set_xticks([0]) 
        axes.flat[i].set_xticklabels([i % matrix_dim])
        axes.flat[i].set_yticks([0]) 
        axes.flat[i].set_yticklabels([floor(i / matrix_dim)])
        axes.flat[i].grid(False)
        # axes.flat[i].xaxis.set_visible(False)
        # axes.flat[i].yaxis.set_visible(False)
        axes.flat[i].label_outer()
        
    fig.subplots_adjust(wspace=0.1, hspace=0.01, right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(sm, cax=cbar_ax)
    plt.suptitle(title, fontsize=20)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

import sys
sys.path.append('./')
from copy import deepcopy
import pickle

import matplotlib.pyplot as plt
from seaborn import heatmap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets import MeasurementDataset, VectorDensityMatrixDataset
from src.model import Regressor, Classifier
from src.torch_utils import test_output_statistics_for_given_feature, test_output_statistics_varying_feature, calculate_dataset_statistics, test_varying_feature, calculate_mean_model_output_with_varied_feature
from src.torch_measure import calculate_concurrence_from_measurements
from src.logging import log_metrics_to_file, plot_metrics_from_file, plot_metrics_from_files

batch_size = 512
test_dataset = MeasurementDataset(root_path='./data/val/')
# test_dataset = VectorDensityMatrixDataset(root_path='./data/val/')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model_name = 'regressor'
results_path_prefix = f'./logs/{model_name}_features_importance/real_distribution_range_features_importance_with_varied_input_mean_'
results_path = '{}{}.log'
plot_path = f'./plots/{model_name}_features_importance/real_distribution_range_features_importance_with_varied_input_mean_' + '{}.png'
cov_matrix_path = f'./plots/{model_name}_features_importance/cov_matrix.png'
model_statistics_path = f'./data/{model_name}_dataset_statistics.pkl'
model_varied_input_mean_path = f'./data/{model_name}_varied_input_mean.pkl'
dataset_statistics_path = './data/dataset_statistics.pkl'

model_path = f'./models/{model_name}.pt'
model_params = {
    'input_dim': 16,
    'output_dim': 1,
    'layers': 2,
    'hidden_size': 128,
    'input_dropout': 0.0
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_name == 'regressor':
    model = Regressor(**model_params)
    model.load(model_path)
    model.eval()
    model.to(device)
elif model_name == 'classifier':
    model = Classifier(**model_params)
    model.load(model_path)
    model.eval()
    model.to(device)
else:
    model = calculate_concurrence_from_measurements


criterions = {
    'mean': lambda x: torch.mean(x).item(),
    'std': lambda x: torch.std(x).item(),
    'max': lambda x: torch.max(x).item(),
    'min': lambda x: torch.min(x).item()
}

try:
    with open(dataset_statistics_path, 'rb') as f:
        dataset_statistics = pickle.load(f)
    mean = dataset_statistics['mean']
    covariance_matrix = dataset_statistics['covariance_matrix']
except:
    dataset_statistics = calculate_dataset_statistics(test_loader, device)
    mean = dataset_statistics['mean']
    covariance_matrix = dataset_statistics['covariance_matrix']
    with open(dataset_statistics_path, 'wb') as f:
        pickle.dump(dataset_statistics, f)
    cov_matrix_plot = heatmap(covariance_matrix.detach().cpu().numpy(), annot=False)
    cov_matrix_plot.get_figure().savefig(cov_matrix_path)
    plt.close()

try:
    with open(model_statistics_path, 'rb') as f:
        model_dataset_statistics = pickle.load(f)
    model_mean = model_dataset_statistics['mean']
    model_covariance_matrix = model_dataset_statistics['covariance_matrix']
except:
    model_dataset_statistics = calculate_dataset_statistics(test_loader, device, model)
    model_mean = model_dataset_statistics['mean']
    model_covariance_matrix = model_dataset_statistics['covariance_matrix']
    with open(model_statistics_path, 'wb') as f:
        pickle.dump(model_dataset_statistics, f)

try:
    with open(model_varied_input_mean_path, 'rb') as f:
        model_varied_input_mean = pickle.load(f)
except:
    model_varied_input_mean = calculate_mean_model_output_with_varied_feature(model, device, test_loader, features_value_range=(0., 1.), step=0.01)
    with open(model_varied_input_mean_path, 'wb') as f:
        pickle.dump(model_varied_input_mean, f)


print("Dataset statistics:")
print("Dataset mean:", mean)
print("Dataset std:", dataset_statistics['std'])
print("Dataset max:", dataset_statistics['max'])
print("Dataset min:", dataset_statistics['min'])

print("Model statistics:")
print("Model mean:", model_mean)
print("Model std:", model_dataset_statistics['std'])
print("Model max:", model_dataset_statistics['max'])
print("Model min:", model_dataset_statistics['min'])

print("Model varied input mean:", model_varied_input_mean)

# for i in range(0, model_params['input_dim']):
#     print('Measurement', i)
#     min_input_value = dataset_statistics['min'][i].item()
#     max_input_value = dataset_statistics['max'][i].item()

#     # dataset_feature_test_metrics = test_output_statistics_for_given_feature(model, device, test_loader, feature_idx=[i], criterions=criterions)
#     # feature_varying_metrics = test_output_statistics_varying_feature(model, device, feature_idx=[i], criterions=criterions, features_num=16, feature_value_range=(min_input_value, max_input_value), step=0.01, mean=mean, covariance_matrix=covariance_matrix)
#     dataset_mean_test_metrics, dataset_distance_test_metrics  = test_varying_feature(model, device, test_loader, criterions, feature_idx=[i], model_output_mean=model_varied_input_mean, feature_value_range=(min_input_value, max_input_value), step=0.01)

#     write_mode = 'w' if i == 0 else 'a'
#     log_metrics_to_file(dataset_mean_test_metrics, results_path.format(results_path_prefix,  'mean_dataset_values'), write_mode=write_mode, xaxis=i, xaxis_name='measurement_idx')        
#     log_metrics_to_file(dataset_distance_test_metrics, results_path.format(results_path_prefix,  'distance_dataset_values'), write_mode=write_mode, xaxis=i, xaxis_name='measurement_idx')        
#     # log_metrics_to_file(feature_varying_metrics, results_path.format(results_path_prefix,  'feature_varying'), write_mode=write_mode, xaxis=i, xaxis_name='measurement_idx')

# # plot_metrics_from_file(results_path.format(results_path_prefix,  'dataset'), title=f'Metrics for all measurements', save_path=plot_path.format(f'dataset'), xaxis='measurement_idx', linestyle='-', marker='x')
# plot_metrics_from_file(results_path.format(results_path_prefix,  'mean_dataset_values'), title=f'Metrics for all measurements', save_path=plot_path.format(f'mean_dataset_values'), xaxis='measurement_idx', linestyle='-', marker='x')
# plot_metrics_from_file(results_path.format(results_path_prefix,  'distance_dataset_values'), title=f'Metrics for all measurements', save_path=plot_path.format(f'distance_dataset_values'), xaxis='measurement_idx', linestyle='-', marker='x')

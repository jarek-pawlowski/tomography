import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm


import typing as t


def calculate_dataset_statistics(
    data_loader: DataLoader,
    device: torch.device,
    callable: t.Optional[t.Callable] = None
):
    # calculate latent space distribution (mean and std)
    population = []
    for data, _ in tqdm(data_loader, 'Collecting data statistics...'):
        data = data.to(device)
        if callable is not None:
            data = callable(data)
        population.append(data)
    population = torch.cat(population, dim=0)
    statistics = {
        'mean': torch.mean(population, dim=0),
        'std': torch.std(population, dim=0),
        'covariance_matrix': torch.cov(population.T),
        'max': torch.max(population, dim=0).values,
        'min': torch.min(population, dim=0).values
    }
    return statistics


def calculate_dataset_histogram(
    data_loader: DataLoader,
    device: torch.device,
    bins: torch.Tensor = torch.tensor([-0.1, 1.e-2, 1.1])
):
    population = []
    for _, label in tqdm(data_loader, 'Collecting data histogram...'):
        label = label.to(device)
        population.append(label)
    population = torch.cat(population, dim=0)
    return torch.histogram(population, bins)[0]


def generate_sample_from_mean_and_covariance(mean: torch.Tensor, covariance_matrix: torch.Tensor, batch_size: int = 1):
    mvn = MultivariateNormal(mean, covariance_matrix)
    return mvn.sample((batch_size,))


def generate_mean_sample(
    data_loader: DataLoader,
    device: torch.device,
):
    samples = [data_tuple[0].to(device) for data_tuple in tqdm(data_loader, 'Collecting data statistics...')]
    samples = torch.cat(samples, dim=0)
    mean_sample = torch.mean(samples, dim=0)
    return mean_sample
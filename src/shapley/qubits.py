#
# library for qubits data manipulation
#
######################################################

import os
import sys

import numpy as np
import pandas as pd
import joblib
import pickle

from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score

######################################################

def filter_strong(df):
    """Filter just unentangled and strongly entangled samples."""
    
    return df[(df["concurrence"] < 1.e-6) | (df["concurrence"] > 0.99)]

######################################################

def get_data(df, selected_features=None, shuffle=True, n_samples=None):
    """Return dataset in NumPy arrays."""
    
    # training samples
    if isinstance(selected_features, list):
        # just selected features
        X = df[selected_features].to_numpy()
    else:
        # all features
        X = df.iloc[:,:16].to_numpy()
        
    Y = df["tag"].to_numpy()          # training tags
    C = df["concurrence"].to_numpy()  # concurrence values
    
    data_size = len(X)  # size of the whole dataset
    
    # if sample size is integer
    if isinstance(n_samples, int):
        if n_samples < data_size:
            # choose random subset of the whole dataset
            Indices = np.random.choice(data_size, n_samples, replace=False)
            X, Y, C = X[Indices], Y[Indices], C[Indices]
                
    # shuffle dataset
    if shuffle:
        Indices = np.random.permutation(len(X))
        X, Y, C = X[Indices], Y[Indices], C[Indices]
    
    return {"x": X, "y": Y, "c": C}
    
######################################################

def load_datasets(Dir, selected_features=None, shuffle=True, strong=False, train_only=False, n_samples=None):
    """Load training and validation datasets."""
    
    data_train = Dir / "data_train.csv"  # training dataset
    data_valid = Dir / "data_valid.csv"  # validation dataset
    
    ### load training dataset
    df_train = pd.read_csv(data_train) 
    
    # filter just unentangled and strongly entangled data
    if strong:
        df_train = filter_strong(df_train)
    
    # get numpy files
    Data_train = get_data(
        df_train, selected_features=selected_features, shuffle=shuffle, n_samples=n_samples
    )

    # get features names
    if isinstance(selected_features, list):
        # just selected features
        Features = selected_features
    else:
        # all features
        Features = df_train.columns.to_list()[:16]  # feature names
    
    # return only training dataset
    if train_only:
        return Data_train, Features
    
    ### load validation dataset
    df_valid = pd.read_csv(data_valid)
    
    # filter just unentangled and strongly entangled data
    if strong:
        df_valid = filter_strong(df_valid)

    # get numpy files
    Data_valid = get_data(
        df_valid, selected_features=selected_features, shuffle=False, n_samples=None
    )
        
    # return training and validation datasets
    return Data_train, Data_valid, Features

######################################################

def get_perturbed_data(data, component_index, mu=0, sigma=1, repeats=1):
    """
    Add Gaussian noise to a specific component of each sample in the data array.

    :param data: NumPy array of shape (n_samples, n_components)
    :param component_index: Index of the component to which noise will be added (0-based).
    :param mu: Mean of the Gaussian noise distribution.
    :param sigma: Standard deviation of the Gaussian noise distribution.
    :param repeats: Number of repeteations of the original dataset
    :return: Modified data array with noise added to the specified component.
    """
    
    # Check if the component_index is within the valid range
    if not (0 <= component_index < data.shape[1]):
        raise ValueError("component_index is out of bounds.")
        
    # Repeat dataset to enlarge statistics
    data_pert = np.repeat(data, repeats=repeats, axis=0)
    
    # Generate Gaussian noise
    noise = np.random.normal(mu, sigma, data_pert.shape[0])
    
    # Add noise to the specified component of each sample
    data_pert[:, component_index] += noise
    
    # Clip the values of the specified component to be between 0 and 1
    data_pert[:, component_index] = np.clip(data_pert[:, component_index], 0, 1)
    
    return data_pert

######################################################

def evaluate_perturbed_data(clf_model, data, tags, component_index, mu=0, sigma=1, repeats=1):
    """
    Evaluate classification model using perturbed data.

    :param clf_model: used classification model
    :param data: NumPy array of shape (n_samples, n_components)
    :param tags: NumPy array of shape (n_samples)
    :param component_index: Index of the component to which noise will be added (0-based).
    :param mu: Mean of the Gaussian noise distribution.
    :param sigma: Standard deviation of the Gaussian noise distribution.
    :param repeats: Number of repeteations of the original dataset
    :return: accuracy score, precision score, recall score.
    """
    
    # Perturbed data
    data_pert = get_perturbed_data(data, component_index, mu=mu, sigma=sigma, repeats=repeats)
    
    # Make predictions
    preds = clf_model.predict(data_pert)
    
    # Prepare tags
    tags_pert = np.repeat(tags, repeats=repeats, axis=0)  

    # Calculate metrics
    accuracy  = accuracy_score(tags_pert, preds)
    precision = precision_score(tags_pert, preds)
    recall    = recall_score(tags_pert, preds)

    return accuracy, precision, recall

######################################################

def get_random_data(data, component_index, repeats=1):
    """
    Generate random numbers between 0 and 1 from a uniform distribution for a specific component 
    of each sample in the data array.

    :param data: NumPy array of shape (n_samples, n_components)
    :param component_index: Index of the component for which to generate random numbers (0-based).
    :param repeats: Number of repeteations of the original dataset
    :return: Modified data array with the specified component replaced by random numbers from a uniform distribution.
    """
    
    # Check if the component_index is within the valid range
    if not (0 <= component_index < data.shape[1]):
        raise ValueError("component_index is out of bounds.")
        
    # Repeat dataset
    data_rand = np.repeat(data, repeats=repeats, axis=0)
    
    # Generate random numbers from a uniform distribution for the specified component
    data_rand[:, component_index] = np.random.uniform(0, 1, data_rand.shape[0])
    
    return data_rand

######################################################

def evaluate_random_data(clf_model, data, tags, component_index, repeats=1):
    """
    Evaluate classification model using perturbed data.

    :param clf_model: used classification model
    :param data: NumPy array of shape (n_samples, n_components)
    :param tags: NumPy array of shape (n_samples)
    :param component_index: Index of the component to which noise will be added (0-based).
    :param mu: Mean of the Gaussian noise distribution.
    :param sigma: Standard deviation of the Gaussian noise distribution.
    :return: accuracy score, precision score, recall score.
    """
    
    # Random data
    data_rand = get_random_data(data, component_index, repeats=repeats)

    # Make predictions
    preds = clf_model.predict(data_rand)
    
    # Prepare tags
    tags_rand = np.repeat(tags, repeats=repeats, axis=0) 

    # Calculate metrics
    accuracy  = accuracy_score(tags_rand, preds)
    precision = precision_score(tags_rand, preds)
    recall    = recall_score(tags_rand, preds)

    return accuracy, precision, recall

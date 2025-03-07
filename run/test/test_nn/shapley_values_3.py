#! /home/balaz/miniconda3/envs/tf/bin/python
#
# calculate shapley values for given
# random forest classifier
#
#################################################

import os
import sys

import numpy as np
import pandas as pd
import joblib
import pickle

from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.shapley.qubits import *
from src.shapley.shapley import *
from src.model import Regressor

#################################################

# Data directory
DATADIR = Path("./data/")

# Results directory
RESDIR = Path("./shapley/")
os.makedirs(RESDIR, exist_ok=True)

MODEL_FILE  = "rf1.joblib"         # File with saved trained model
MODEL_LABEL = "regressor"                # Model label
MODEL_PATH  = RESDIR / MODEL_FILE  # Path to saved trained RF model

#################################################

N_BG_SAMPLES   = 100    # number of background samples for Shaple
N_SAMPLES      = 1000   # number of samples in the dataset for calculation of shapley values
N_PERMUTATIONS = 100    # Number of permutations in Shapley values calculations
STRONG         = False  # unentangled and strongly entangled samples only

#################################################

## Load data
print("Loading datasets ...")

Data_train, Data_valid, Features = load_datasets(DATADIR, strong=STRONG, n_samples=None)

X_train, Y_train, C_train = Data_train["x"], Data_train["y"], Data_train["c"]
X_valid, Y_valid, C_valid = Data_valid["x"], Data_valid["y"], Data_valid["c"]

## Print info
print("Dataset loaded.")
print(f"Number of training instances: {len(X_train):d}")
print(f"Number of test instances: {len(X_valid):d}")

## load model
# print("Loading RF model ...")

# model = joblib.load(MODEL_PATH)

# print("RF model loaded.")

model_name = 'regressor'
model_path = f'./{model_name}.pt'
model_params = {
    'input_dim': 16,
    'output_dim': 1,
    'layers': 2,
    'hidden_size': 128,
    'input_dropout': 0.
}
model = Regressor(**model_params)
model.load(model_path, map_location='cpu')

#################################################
## SHAPLEY VALUES

print("Calculating Shapley values ...")

# Choose backround samples dataset
rng = np.random.default_rng(42)
bg_indices = rng.choice(len(X_train), size=N_BG_SAMPLES, replace=False)
X_bg = X_train[bg_indices]
print(f"X_bg shape: {X_bg.shape}")

# Choose samples for shapley explanation
test_indices = rng.choice(len(X_valid), size=N_SAMPLES, replace=False)
X_sh = X_valid[test_indices]  # samples used for Shapley values calculations
print(f"X_sh shape: {X_sh.shape}")

# List to store Shapley values
all_shap_vals = []

# Calculate shap values for few samples from test dataset
for x in tqdm(X_sh):
    phi = approximate_shap_values(model, x, X_bg, n_permutations=N_PERMUTATIONS, class_idx=0)
    all_shap_vals.append(phi)

# Calculate global Shapley values
all_shap_vals = np.array(all_shap_vals)             # shape (N, M)
global_shap_vals = np.mean(all_shap_vals, axis=0)   # Average Shapley values - shape (M,)

print("Done.")
print("Global average Shapley values across test samples:\n", global_shap_vals)

## save shapley values
file_name = RESDIR / f"{model_name}_shap_values_class0.pickle"
with open(file_name, "wb") as file:
    pickle.dump(global_shap_vals, file)

print(f"Shapley valuse saved to:\n{file_name}")

#! /home/balaz/miniconda3/envs/tf/bin/python
#
# random forest classifier
#
#################################################

import os
import sys

import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

#################################################

### data location
DATADIR = Path("./Data/")

DATA_TRAIN = DATADIR / "data_train.csv"
DATA_VALID = DATADIR / "data_valid.csv"

TAG = "tag"

### random forest
NUM_ESTIM = 1000    # number of classifiers
MAX_FEAT  = None    # number of features in each classifier
MAX_DEPTH = 20      # maximum depth
CRIT      = "gini"  # criterion

BOOTSTRAP = True    # do bootstraping
OOB_SCORE = True    # calculate oob-score

#################################################
### data preparation

# load training dataset
df_train = pd.read_csv(DATA_TRAIN)
NUM_DATA = len(df_train)  # number of samples in the dataset

X_train = df_train.iloc[:,:16].to_numpy()  # training samples
Y_train = df_train[TAG].to_numpy()          # training tags

#################################################
### random forest classifier

# create model
clf = RandomForestClassifier(
    n_estimators=NUM_ESTIM, max_features=MAX_FEAT, criterion=CRIT, 
    max_depth=MAX_DEPTH,
    bootstrap=BOOTSTRAP, oob_score=OOB_SCORE,
    verbose=0
)

# fit model
clf.fit(X_train, Y_train)

#################################################
### save model

mfilename = "rf_maxdepth20.joblib"
joblib.dump(clf, mfilename, compress=0)

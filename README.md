# INSY-7120_final_project
This repository contains the final project for INSY-7120.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

#load data
HDDP_data_og = pd.read_csv(r"C:\Users\jalaw\Downloads\HD smaller (1).csv")
#clean up features
#turn names to numbers
HDDP_clean = pd.get_dummies(HDDP_data_og, columns=['REGISTER_LOCATION', 'HH_SEGMENT'])
#Fill all NaNs with medians b/c no other way to use that data otherwise (not enough data to drop the nas)
HDDP_clean = HDDP_clean.fillna(HDDP_data_og.median(numeric_only=True))
#drop the id number column (use axis = 1)  because it's just a label
HDDP_clean = HDDP_clean.drop(HDDP_clean.columns[0], axis=1)

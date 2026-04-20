# INSY-7120_final_project
This repository contains the final project for INSY-7120.

#lin reg/dummy/regularization imports
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
HDDP_clean = pd.get_dummies(HDDP_data_og, columns=['REGISTER_LOCATION', 'HH_SEGMENT'], drop_first=True)
#Fill all NaNs with medians b/c no other way to use that data otherwise (not enough data to drop the nas)
HDDP_clean = HDDP_clean.fillna(HDDP_data_og.median(numeric_only=True))
#drop the id number column (use axis = 1)  because it's just a label
HDDP_clean = HDDP_clean.drop(HDDP_clean.columns[0], axis=1)

#----------------------------------------------------------------------------------------------
# 1. Identify columns you want to get rid of across the board
# We'll drop anything that has 'SLS' or 'QTY' in the name, 
# UNLESS it also has 'ELIGIBLE' in it.
cols_to_drop = [col for col in HDDP_clean.columns 
                if ('_SLS' in col or '_QTY' in col) 
                and 'ELIGIBLE' not in col]

# 2. Execute the drop
HDDP_noncolinear = HDDP_clean.drop(columns=cols_to_drop)


# --- NEW ADDITION: Manually drop the final redundant pairs ---
manual_drops = [
    'R52_CUST_TXNS', 
    'R52_CUST_SALES_AMT', 
    'REGISTER_LOCATION_MAINLINE'
]
# We add errors='ignore' just in case you already dropped one by accident
HDDP_noncolinear = HDDP_noncolinear.drop(columns=manual_drops, errors='ignore')

# 3. Check what's left
print("Features remaining:", HDDP_noncolinear.columns.tolist())
print(f"Total features remaining: {len(HDDP_noncolinear.columns)}")

#----------------------------------------------------------------------------------------------
#look at correlations for bad features to drop
correlations = HDDP_noncolinear.corr()
correlations_rounded = correlations.round(3)
# 2. Sort the correlations for your target variable
# Replace 'HDDP_PURCHASED_FLAG' with the exact column name if different
target_corr = correlations['HDPP_PURCHASED_FLAG'].sort_values(ascending=False)
# 3. Print the sorted list
print("Correlation with Purchase Flag (Highest to Lowest):")
print(target_corr.round(3))

#----------------------------------------------------------------------------------------------
# 1. Take the absolute value of the correlation matrix
abs_corr = correlations.abs()

# 2. Unstack it into a list of pairs
sol = abs_corr.unstack()

# 3. Sort by value and remove the ones that correlate with themselves (1.0)
sorted_pairs = sol.sort_values(kind="quicksort", ascending=False)
strong_pairs = sorted_pairs[sorted_pairs < 1]

print("Strongest Relationships (Absolute Value):")
print(strong_pairs.head(30))
#----------------------------------------------------------------------------------------------
#-----------------------COLINEARITY FINALLY GONE-----------------------------------------------
#----------------------------------------------------------------------------------------------
#start making model

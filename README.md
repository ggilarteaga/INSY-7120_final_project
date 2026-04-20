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
HDPP_data_og = pd.read_csv(r"C:\Users\jalaw\Downloads\HD smaller (1).csv")
#clean up features
#turn names to numbers
HDPP_clean = pd.get_dummies(HDPP_data_og, columns=['REGISTER_LOCATION', 'HH_SEGMENT'], drop_first=True)
#Fill all NaNs with medians b/c no other way to use that data otherwise (not enough data to drop the nas)
HDPP_clean = HDPP_clean.fillna(HDPP_data_og.median(numeric_only=True))
#drop the id number column (use axis = 1)  because it's just a label
HDPP_clean = HDPP_clean.drop(HDPP_clean.columns[0], axis=1)

#----------------------------------------------------------------------------------------------
# 1. Identify columns you want to get rid of across the board
# We'll drop anything that has 'SLS' or 'QTY' in the name, 
# UNLESS it also has 'ELIGIBLE' in it.
cols_to_drop = [col for col in HDPP_clean.columns 
                if ('_SLS' in col or '_QTY' in col) 
                and 'ELIGIBLE' not in col]

# 2. Execute the drop
HDPP_noncolinear = HDPP_clean.drop(columns=cols_to_drop)


# --- NEW ADDITION: Manually drop the final redundant pairs ---
manual_drops = [
    'R52_CUST_TXNS', 
    'R52_CUST_SALES_AMT', 
    'REGISTER_LOCATION_MAINLINE'
]
# We add errors='ignore' just in case you already dropped one by accident
HDPP_noncolinear = HDPP_noncolinear.drop(columns=manual_drops, errors='ignore')

# 3. Check what's left
#print("Features remaining:", HDPP_noncolinear.columns.tolist())
print(f"Total features remaining: {len(HDPP_noncolinear.columns)}")

#----------------------------------------------------------------------------------------------
#look at correlations for bad features to drop
correlations = HDPP_noncolinear.corr()
correlations_rounded = correlations.round(3)
# 2. Sort the correlations for your target variable
# Replace 'HDPP_PURCHASED_FLAG' with the exact column name if different
target_corr = correlations['HDPP_PURCHASED_FLAG'].sort_values(ascending=False)
# 3. Print the sorted list
#print("Correlation with Purchase Flag (Highest to Lowest):")
#print(target_corr.round(3))

#----------------------------------------------------------------------------------------------
# 1. Take the absolute value of the correlation matrix
abs_corr = correlations.abs()

# 2. Unstack it into a list of pairs
sol = abs_corr.unstack()

# 3. Sort by value and remove the ones that correlate with themselves (1.0)
sorted_pairs = sol.sort_values(kind="quicksort", ascending=False)
strong_pairs = sorted_pairs[sorted_pairs < 1]

#print("Strongest Relationships (Absolute Value):")
#print(strong_pairs.head(30))
#----------------------------------------------------------------------------------------------
#-----------------------COLINEARITY FINALLY GONE-----------------------------------------------
#----------------------------------------------------------------------------------------------
#define vars
X = HDPP_noncolinear.drop(columns=["HDPP_PURCHASED_FLAG"])
y = HDPP_noncolinear["HDPP_PURCHASED_FLAG"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {X_train.shape[0]} observations, {X_train.shape[1]} features")
print(f"Test set:     {X_test.shape[0]} observations")

#dummy
dummy = make_pipeline(DummyRegressor(strategy='mean'))
dummy.fit(X_train, y_train)
print(f"Train R²: {dummy.score(X_train, y_train):.4f}")
print(f"Test R²:  {dummy.score(X_test, y_test):.4f}")

#straight up linear regression
LinRegModel = make_pipeline(StandardScaler(), LinearRegression())
LinRegModel.fit(X_train, y_train)

print("\nNEED CV")
print(f"Train R²: {LinRegModel.score(X_train, y_train):.4f}")
print(f"Test R²:  {LinRegModel.score(X_test, y_test):.4f}")

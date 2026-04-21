# INSY-7120_final_project
This repository contains the final project for INSY-7120.

#lin reg/dummy/regularization imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

#---------------------LOAD & CLEAN DATA----------------------------------------
#load data
HDPP_data_og = pd.read_csv(r"C:\Users\jalaw\Downloads\HD smaller (1).csv")

#clean up features
#turn names to numbers
HDPP_clean = pd.get_dummies(HDPP_data_og, columns=['REGISTER_LOCATION', 'HH_SEGMENT'], drop_first=True)

#Fill all NaNs with medians b/c no other way to use that data otherwise (not enough data to drop the nas)
HDPP_clean = HDPP_clean.fillna(HDPP_data_og.median(numeric_only=True))

#drop the id number column (use axis = 1)  because it's just a label
HDPP_clean = HDPP_clean.drop(HDPP_clean.columns[0], axis=1)

#------------------------REMOVE MULTICOLINEARITY-------------------------------
#Check for multicolinearity by looking at correlations
clean_corr = HDPP_clean.corr().abs()

# Unstack into pairs and remove self-correlations
clean_pairs = (
    clean_corr
    .unstack()
    .sort_values(ascending=False)
)
clean_pairs = clean_pairs[clean_pairs < 1]  # drop self-correlations (1.0)



print("=== Correlation Pairs in HDPP_clean (Before Dropping) ===")
print(clean_pairs.head(60).round(3))  

# drop anything that has 'SLS' or 'QTY' in the name, UNLESS it also has 'ELIGIBLE' in it
# b/c those relationships all have high correlations and are logically similar categorically, 
# and we only want to keep 1 (eligible)
cols_to_drop = [col for col in HDPP_clean.columns 
                if ('_SLS' in col or '_QTY' in col) 
                and 'ELIGIBLE' not in col]

# Execute the drop
HDPP_noncolinear = HDPP_clean.drop(columns=cols_to_drop)

# Drop these too, as they also are redundant upon inspection of correlations
manual_drops = [
    'R52_CUST_TXNS', 
    'R52_CUST_SALES_AMT', 
    'REGISTER_LOCATION_MAINLINE'
]
# We add errors='ignore' just in case you already dropped one by accident
HDPP_noncolinear = HDPP_noncolinear.drop(columns=manual_drops, errors='ignore')

#Check what's left
#print("Features remaining:", HDPP_noncolinear.columns.tolist())
print(f"Total features remaining: {len(HDPP_noncolinear.columns)}")

#recheck highest correlations and DOUBLE check for patterns
double_corr = HDPP_noncolinear.corr().abs()
clean_pairs2 = (
    double_corr
    .unstack()
    .sort_values(ascending=False)
)
clean_pairs2 = clean_pairs2[clean_pairs2 < 1]  # drop self-correlations (1.0)



print("=== Correlation Pairs in HDPP_clean (After Dropping) ===")
print(clean_pairs2.head(10).round(3))  

#look at cleaned correlations for correlations to response var (checking leakage)
correlations = HDPP_noncolinear.corr()
correlations_rounded = correlations.round(3)
# Sort the correlations for your target variable
target_corr = correlations['HDPP_PURCHASED_FLAG'].sort_values(ascending=False)
# Print the sorted list
print("Correlation with Purchase Flag (Highest to Lowest):")
print(target_corr.round(3))

#------------------------------------------------------------------------------
#-----------------------START BUILDING MODELS----------------------------------
#------------------------------------------------------------------------------
#define vars, tts
X = HDPP_noncolinear.drop(columns=["HDPP_PURCHASED_FLAG"])
y = HDPP_noncolinear["HDPP_PURCHASED_FLAG"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {X_train.shape[0]} observations")
print(f"Test set:     {X_test.shape[0]} observations")

#---------------------------DUMMY----------------------------------------------
dummy = make_pipeline(DummyRegressor(strategy='mean'))
dummy.fit(X_train, y_train)
print("\nDummy")
print(f"Train R²: {dummy.score(X_train, y_train):.4f}")
print(f"Test R²:  {dummy.score(X_test, y_test):.4f}")

#---------------------BASIC LINEAR REGRESSION (OLS)----------------------------
OLSModel = make_pipeline(StandardScaler(), LinearRegression())
OLSModel.fit(X_train, y_train)

# Create a CV splitter with explicit shuffle control -> make it stratified since there are few insurance purchases
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
OLSscores = cross_val_score(OLSModel, X_train, y_train, cv=kf, scoring='r2')

print("\nOLS")
print(f"Train R²: {OLSModel.score(X_train, y_train):.4f}")
print(f"CV R²:  {OLSscores.mean():.4f} (+/- {OLSscores.std():.4f})")

#------------------------------RIDGE-------------------------------------------
print("\nRidge")
# Find best alphas
alphas = np.logspace(-3, 3, 100)  # test 100 values from 0.001 to 1000
ridge_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=kf))
ridge_cv.fit(X_train, y_train)
best_alpha_ridge = ridge_cv.named_steps['ridgecv'].alpha_
print(f"Best Ridge alpha: {best_alpha_ridge:.4f}")

#make and fit optimal ridge
RidgeModel = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ridge))
RidgeModel.fit(X_train, y_train)

#score
Ridgescores = cross_val_score(RidgeModel, X_train, y_train, cv=kf, scoring='r2')
print(f"Train R²: {RidgeModel.score(X_train, y_train):.4f}")
print(f"CV R²:  {Ridgescores.mean():.4f} (+/- {Ridgescores.std():.4f})")

#------------------------------LASSO-------------------------------------------
print("\nLasso")
# find alphas again
lasso_cv = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=kf, max_iter=10000))
lasso_cv.fit(X_train, y_train)
best_alpha_lasso = lasso_cv.named_steps['lassocv'].alpha_
print(f"Best Lasso alpha: {best_alpha_lasso:.4f}")

#make and fit optimal lasso
LassoModel = make_pipeline(StandardScaler(), Lasso(alpha=best_alpha_lasso, max_iter=10000))
LassoModel.fit(X_train, y_train)

#score
Lassoscores = cross_val_score(LassoModel, X_train, y_train, cv=kf, scoring='r2')
print(f"Train R²: {LassoModel.score(X_train, y_train):.4f}")
print(f"CV R²:  {Lassoscores.mean():.4f} (+/- {Lassoscores.std():.4f})")

#--------------------------LOGISTIC REGRESSION---------------------------------



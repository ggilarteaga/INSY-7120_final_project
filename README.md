# INSY-7120_final_project
This repository contains the final project for INSY-7120.

#lin reg/dummy/regularization/logreg imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

#---------------------LOAD & CLEAN DATA----------------------------------------
#load data
HDPP_data_og = pd.read_csv(r"C:\Users\jalaw\Downloads\HD smaller (1).csv")

#clean up features
#turn names to numbers
HDPP_clean = pd.get_dummies(HDPP_data_og, columns=['REGISTER_LOCATION', 'HH_SEGMENT'], drop_first=True)

#filling in Nans here instead of in pipeline with simpleimputer because
#I've already set up code to be ordered strictly clean, drop, model
#process: make new column indicating if there's missing info, then go back and 
#fill missing info with 0, then look at correlatinos of new and original features

#IDENTIFY columns missing a lot of data
cols_to_flag = ['R52_CUST_SALES_AMT'] 

#CREATE flags for missing data
for col in cols_to_flag:
    if col in HDPP_clean.columns:
        # Create 1/0 indicator column
        HDPP_clean[f'{col}_IS_MISSING'] = HDPP_clean[col].isna().astype(int)

# fill the original columns' missing data with 0's; the flags take care of the zeroes' importance
HDPP_clean['R52_CUST_SALES_AMT'] = HDPP_clean['R52_CUST_SALES_AMT'].fillna(0)
HDPP_clean['R52_CUST_HDPP_PURCHASED_PCT'] = HDPP_clean['R52_CUST_HDPP_PURCHASED_PCT'].fillna(0)
HDPP_clean['R52_CUST_HDPP_ELIGIBLE_TXNS'] = HDPP_clean['R52_CUST_HDPP_ELIGIBLE_TXNS'].fillna(0)
HDPP_clean['R52_CUST_TXNS'] = HDPP_clean['R52_CUST_TXNS'].fillna(0)

# Now do  median fill for everything else missing
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
                and 'ELIGIBLE' not in col and 'SALES_AMT' not in col]

# Execute the drop
HDPP_noncolinear = HDPP_clean.drop(columns=cols_to_drop)

# Drop these too, as they also are redundant upon inspection of correlations
#keep R52 sales over eligible transactions since it's associated with the missing data column
manual_drops = [
    'REGISTER_LOCATION_MAINLINE', 
    'R52_CUST_TXNS',
    'R52_CUST_HDPP_ELIGIBLE_TXNS'
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
kfstrat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfreg = KFold(n_splits=5, shuffle=True, random_state=42)

# Custom scorer that uses predict() instead of predict_proba()
roc_auc_reg_scorer = make_scorer(roc_auc_score)
OLSscores = cross_val_score(OLSModel, X_train, y_train, cv=kfreg, scoring='r2')

print("\nOLS")
print(f"Train R²: {OLSModel.score(X_train, y_train):.4f}")
print(f"CV R²:  {OLSscores.mean():.4f} (+/- {OLSscores.std():.4f})")

#OLS
OLSScoresSt = cross_val_score(OLSModel, X_train, y_train, cv=kfreg, scoring=roc_auc_reg_scorer)
OLS_train_auc = roc_auc_score(y_train, OLSModel.predict(X_train))
print(f"Train AUC: {OLS_train_auc:.4f}")
print(f"CV AUC:    {OLSScoresSt.mean():.4f} (+/- {OLSScoresSt.std():.4f})")





#------------------------------RIDGE-------------------------------------------
print("\nRidge")
# Find best alphas
alphas = np.logspace(-3, 3, 100)  # test 100 values from 0.001 to 1000
ridge_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=kfreg))
ridge_cv.fit(X_train, y_train)
best_alpha_ridge = ridge_cv.named_steps['ridgecv'].alpha_
print(f"Best Ridge alpha: {best_alpha_ridge:.4f}")

#make and fit optimal ridge
RidgeModel = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha_ridge))
RidgeModel.fit(X_train, y_train)

#score
Ridgescores = cross_val_score(RidgeModel, X_train, y_train, cv=kfreg, scoring='r2')
print(f"Train R²: {RidgeModel.score(X_train, y_train):.4f}")
print(f"CV R²:  {Ridgescores.mean():.4f} (+/- {Ridgescores.std():.4f})")

# Ridge
RidgeScoresSt = cross_val_score(RidgeModel, X_train, y_train, cv=kfreg, scoring=roc_auc_reg_scorer)
ridge_train_auc = roc_auc_score(y_train, RidgeModel.predict(X_train))
print(f"Train AUC: {ridge_train_auc:.4f}")
print(f"CV AUC:    {RidgeScoresSt.mean():.4f} (+/- {RidgeScoresSt.std():.4f})")

#------------------------------LASSO-------------------------------------------
print("\nLasso")
# find alphas again
lasso_cv = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=kfreg, max_iter=10000))
lasso_cv.fit(X_train, y_train)
best_alpha_lasso = lasso_cv.named_steps['lassocv'].alpha_
print(f"Best Lasso alpha: {best_alpha_lasso:.4f}")

#make and fit optimal lasso
LassoModel = make_pipeline(StandardScaler(), Lasso(alpha=best_alpha_lasso, max_iter=10000))
LassoModel.fit(X_train, y_train)

#score
Lassoscores = cross_val_score(LassoModel, X_train, y_train, cv=kfreg, scoring='r2')
print(f"Train R²: {LassoModel.score(X_train, y_train):.4f}")
print(f"CV R²:  {Lassoscores.mean():.4f} (+/- {Lassoscores.std():.4f})")

# Lasso
LassoScoresSt = cross_val_score(LassoModel, X_train, y_train, cv=kfreg, scoring=roc_auc_reg_scorer)
lasso_train_auc = roc_auc_score(y_train, LassoModel.predict(X_train))
print(f"Train AUC: {lasso_train_auc:.4f}")
print(f"CV AUC:    {LassoScoresSt.mean():.4f} (+/- {LassoScoresSt.std():.4f})")

#--------------------------LOGISTIC REGRESSION---------------------------------
print("\nLogReg")

LogRegModel = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
)
LogRegModel.fit(X_train, y_train)

LogRegScores = cross_val_score(LogRegModel, X_train, y_train, cv = kfstrat, scoring = 'roc_auc')
logreg_train_auc = roc_auc_score(y_train, LogRegModel.predict_proba(X_train)[:,1])
print(f"Train roc AUC: {logreg_train_auc:.4f}")
print(f"CV roc auc:  {LogRegScores.mean():.4f} (+/- {LogRegScores.std():.4f})")

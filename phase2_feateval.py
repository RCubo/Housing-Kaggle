""" 
Code corresponding to a submission to Kaggle's housing competition for Python 3
Libraries used:
    numpy
    matplotlib
    scikit-learn
    pandas
    XGBoost    
    seaborn (optional)

Performance: 0.1311 in Kaggle's test set

(C) 2017 Ruben Cubo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from xgboost import XGBRegressor

def data_init():
    
    #Reads the .csv files in order to initialize the data
    
    house_train = pd.read_csv('train.csv')
    
    house_test = pd.read_csv('test.csv')
    
    return house_train, house_test

def score_calc_log(features, labels, regressor):
    
    # Computes the score from a regressor
    
    labels_pred = regressor.predict(features)
    err_pred = mean_squared_error(np.log(labels), np.log(labels_pred))**0.5
    return err_pred

def score_calc_log_labels(labels_true, labels_pred):
    
    # Computes the score directly from the labels (similar to the one above)
    err_pred = mean_squared_error(np.log(labels_true), np.log(labels_pred))**0.5
    return err_pred

# Load the data
house_train, house_test = data_init()

# MSSubclass is a categorical feature with numerical values.
# The easiest way is to transform it into a string.
house_train['MSSubClass'] = house_train['MSSubClass'].astype(str)
house_test['MSSubClass'] = house_test['MSSubClass'].astype(str)

# Join both datasets for feature cleaning and encoding
house_all = pd.concat([house_train, house_test])
# We take out the values with the most NaNs
house_all = house_all.drop(['FireplaceQu','Fence','Alley','MiscFeature','PoolQC','GarageYrBlt','GarageType','LotFrontage'], axis=1)

# One hot encoding for categorical variables
house_all = pd.get_dummies(house_all)

# We split again the (clean) variables
house_train = house_all.iloc[0:1460,:]
house_test = house_all.iloc[1460:,:]
# For the training set, we just delete all rows that has NaNs
house_train = house_train.dropna()
# For the testing set however, we need to keep everything, so we just put them to 0
house_test = house_test.drop('SalePrice',axis=1)
house_test = house_test.fillna(0)

# We separate now the labels
labels_train = house_train['SalePrice']

# Feature selection

# First, we see the correlations

house_train_corr = house_train.corr()
house_train_corr = house_train_corr.iloc[28,:] #We reduce it to a vector
house_train_corr = house_train_corr.drop('SalePrice')
house_train_corr = np.abs(house_train_corr) #We are only interested in their absolute value

# We sort them in descending order
house_train_corr = house_train_corr.sort_values(ascending=False)

# We plot them in a bar plot
plt.bar(range(0,len(house_train_corr)),house_train_corr.values)
plt.xticks(range(0,len(house_train_corr)),house_train_corr.keys(),rotation='vertical')
plt.show()

# We select 250 features (it can be extended to various values)
features_keys = house_train_corr.keys()
feat_length = np.arange(250, 270, 30)
err_features = []

for k in feat_length:
    
    # We pick the best features
    keys_id = np.arange(1,k)
    features_train = house_train[features_keys[keys_id]]
    features_test = house_test[features_keys[keys_id]]

    # We now normalize them between 0 and 1

    features_scaler = MinMaxScaler()
    features_train_norm = features_scaler.fit_transform(features_train)
    features_test_norm = features_scaler.transform(features_test)

    # We now fit the model
    
    # In this case, we use CV to determine the maximum depth of a decision tree
    
    cv_sets = ShuffleSplit(features_train_norm.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    
    # Define the XGBoost regressor
    
    xgboost_regressor = XGBRegressor(n_estimators=100)
    
    # We take the values we want to look into. In this case, the maximum depth and the learning rate
    params = {'max_depth': [3,4,5,6,7,10,15,20], 'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]}
    
    # We define our scoring function. We want it to be as low as possible!
    scoring_fcn = make_scorer(score_calc_log_labels, greater_is_better=False)
    
    # We define the Grid Search with Cross Validation and we fit it
    grid = GridSearchCV(xgboost_regressor,params,scoring_fcn,cv=cv_sets)
    grid.fit(features_train_norm,labels_train)
    
    print(grid.grid_scores_) # We print the whole score set of the different value
    print('Score for', k, 'features is:', score_calc_log(features_train_norm, labels_train, grid.best_estimator_))
    
    err_features.append(score_calc_log(features_train_norm, labels_train, grid.best_estimator_))

# We predict now the values for the testing set    
labels_test = grid.best_estimator_.predict(features_test_norm)

# We format it according to Kaggle's standards and export it to .csv
test_output = pd.DataFrame(np.squeeze(labels_test), columns=['SalePrice'])
test_output['Id'] = house_test['Id']
test_output.to_csv(path_or_buf = 'testOutputv7_xgbcv.csv',
                   columns = ['Id', 'SalePrice'], index = False)
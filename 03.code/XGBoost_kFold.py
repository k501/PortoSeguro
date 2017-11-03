
# coding: utf-8

# In[25]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb


# In[26]:

#Define the gini metric
# from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score


# In[27]:

# データ読み込み
train = pd.read_csv('01.data/train.csv')
test = pd.read_csv('01.data/test.csv') 


# In[29]:

# 特徴量と目的変数を分離
features = train.drop(['id','target'], axis=1).values
targets = train.target.values


# In[33]:

# Drop unnessesary column
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
unwanted


# In[34]:

# 利用しない変数を除外
train = train.drop(unwanted, axis=1)  
test = test.drop(unwanted, axis=1)  


# In[35]:

# KFoldの設定
kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)


# In[36]:

# XGBoost 
# More parameters has to be tuned. Good luck :)
params = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700
    }


# In[43]:

# Define X , y
X = train.drop(['id', 'target'], axis=1).values
y = train.target.values
test_id = test.id.values
test = test.drop('id', axis=1)

sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)


# In[44]:

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_valid, y_valid)
    d_test = xgb.DMatrix(test.values)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # Train the model! We pass in a max of 2,000 rounds (with early stopping after 100)
    # and the custom metric (maximize=True tells xgb that higher metric is better)
    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    # Predict on our test data
    p_test = mdl.predict(d_test)
    sub['target'] += p_test/kfold


# In[46]:

sub.to_csv('02.output/StratifiedKFold.csv', index=False)


# In[ ]:




# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:03:24 2016

@author: ur57
"""

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

def rmse(y, y_pred):    
    mse = mean_squared_error(y, y_pred)
    return mse ** 0.5

               
if __name__ == '__main__':
        
    y_train = joblib.load('y_train.pkl')
    num_train = joblib.load('num_train.pkl')
    df_val_metrics = joblib.load('df_val_metrics.pkl')
    df_train = df_val_metrics[:num_train] 
    df_test = df_val_metrics[num_train:] 
    
        
    params_bagr = {'learning_rate': [0.1, 0.01],
                   'n_estimators': [100, 500, 1000],
                   'max_depth': [3, 6 ,9],                    
                   'subsample': [0.5, 1.0],
               }
               
    gbr = GradientBoostingRegressor()
    rmse_scorer = make_scorer(rmse, 
                              greater_is_better=False,
                              )
    
    grid = GridSearchCV(gbr, 
                        param_grid=params_bagr, 
                        cv=5,
                        verbose=3,
                        n_jobs=4,
                        scoring=rmse_scorer)

    grid.fit(df_train, y_train)  
    
    print("Best parameters found by grid search:")
    print(grid.best_params_)
    print("Best CV score:")
    print(grid.best_score_)             
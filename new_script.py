# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:03:24 2016

@author: ur57
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
from sklearn.cross_validation import train_test_split
from itertools import combinations

datasets = ['df_char.pkl',
            'df_word.pkl',
            'df_gram.pkl',
            'df_edit.pkl',
            'df_char_nstem.pkl',
            'df_word_nstem.pkl',
            'df_gram_nstem.pkl',
            'df_edit_nstem.pkl',
            'df_tdif.pkl',
            'df_w2v.pkl',
            'df_tdif_nstem.pkl',
            'df_w2v_nstem.pkl',
            ]


def rmse(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    return mse ** 0.5

rmse_scorer = make_scorer(rmse,
                          greater_is_better=False,
                          )


def base_grid_search(est, x, y, params, fit_params=None):

    grid = GridSearchCV(est,
                        params,
                        verbose=3,
                        scoring=rmse_scorer,
                        error_score=-100,
                        n_jobs=24,
                        fit_params=fit_params,
                        cv=5,
                        refit=False
                        )

    grid.fit(x, y)
    return grid


def base_cross_val(est, x, y, fit_params=None, n_jobs=5):
    res = cross_val_score(est,
                          x,
                          y,
                          cv=5,
                          scoring=rmse_scorer,
                          verbose=3,
                          n_jobs=n_jobs,
                          fit_params=fit_params,
                          )
    print(np.mean(res), np.std(res))


def find_best_features(df_train, y_train):
    rfr = RandomForestRegressor(n_estimators=500,
                                max_depth=6,
                                n_jobs=16)

    # vals_pearson = df_train.corr('pearson').values
    vals_pearson = joblib.load('vals_pearson.pkl')
    # vals_kendall = df_train.corr('kendall').values
    # vals_spearman = df_train.corr('spearman').values
    vals_spearman = joblib.load('vals_spearman.pkl')

    vals = (vals_pearson + vals_spearman) / 2

    dumped_cols = []
    res_cols = [True] * vals.shape[0]
    for i in range(vals.shape[0]):
        if i not in dumped_cols:
            for j in range(vals.shape[1]):
                if i != j:
                    if abs(vals[i, j]) > 0.90:
                        dumped_cols.append(j)
                        res_cols[j] = False

    #df_train2 = df_train[df_train.columns[res_cols]]

    rfecv = RFECV(rfr,
                  step=10,  # Float step gives error on the end
                  cv=5,
                  scoring=rmse_scorer,
                  verbose=2)

    # rfecv.fit(df_train2, y_train)
    rfecv = joblib.load('rfecv.pkl')

    return (res_cols, rfecv.get_support())


class MyXGBR(XGBRegressor):
    @property
    def feature_importances_(self):
        """
        Returns
        -------
        feature_importances_ : array of shape = [n_features]
        """
        fs = self.booster().get_fscore()
        keys = [int(k.replace('f', '')) for k in fs.keys()]
        fs_dict = dict(zip(keys, fs.values()))
        all_features_dict = dict.fromkeys(range(0, self._features_count), 0)
        all_features_dict.update(fs_dict)
        all_features = np.fromiter(all_features_dict.values(), np.float32)
        return all_features / all_features.sum()


    def fit(self, X, y, eval_set=None, eval_metric=None, early_stopping_rounds=None, verbose=True):
        self._features_count = X.shape[1]
        super(XGBRegressor, self).fit(X, y, eval_set, eval_metric, early_stopping_rounds, verbose)
        return self

if __name__ == '__main__':

    y_train = joblib.load('y_train.pkl')
    y_train = y_train.astype('float32', 'C')
    id_test = joblib.load('id_test.pkl')

    # num_train = joblib.load('num_train.pkl')
    # df_val_metrics = joblib.load('df_val_metrics.pkl')

    # df_train = df_val_metrics[:num_train]
    # df_test = df_val_metrics[num_train:]
    # del df_val_metrics

    # uncorr_cols, rfecv_cols = find_best_features(df_train, y_train)
    # df_train1 = df_train[df_train.columns[uncorr_cols]]
    # df_train2 = df_train1[df_train1.columns[rfecv_cols]]

    # df_test1 = df_test[df_train1.columns]
    # df_test2 = df_test[df_train2.columns]

    params_xgb = {
                  ### First param tunning
                  # 'learning_rate': [0.05, 0.1, 0.15],
                  # 'n_estimators': np.linspace(200, 1000, 5, dtype='int'),
                  ### Second param tunning
                  # 'max_depth': np.linspace(8, 15, 5, dtype='int32'),
                  # 'min_child_weight': np.linspace(200, 250, 5, dtype='int32'),
                  ### Third param tunning
                  # 'gamma': np.linspace(0.0, 0.5, 5),
                  ### Fourth param tunning
                  # 'subsample': np.linspace(0.6, 0.9, 4),
                  # 'colsample_bytree': np.linspace(0.6, 0.9, 4),
                  ### Fifith param tunning
                  # 'reg_alpha': np.linspace(1e-5, 100, 5)
                  }

    xgbr = XGBRegressor(nthread=25,
                        seed=42,
                        learning_rate=0.02,
                        n_estimators=3000,
                        max_depth=11,
                        min_child_weight=225,
                        gamma=0.125,
                        colsample_bytree=0.6,
                        subsample=0.9,
                        reg_alpha=25,
                        )
    '''
    fit_params_xgb = {'eval_metric': 'rmse',
                      'early_stopping_rounds': 30,
                      'verbose': False,
                      'eval_set': [(X_test, y_test)],
                      }

    bag = BaggingRegressor(xgbr,
                           n_estimators=5,
                           max_samples=0.85,
                           n_jobs=5,
                           random_state=42)

    grid = base_grid_search(xgbr,
                            X_train,
                            y_train,
                            params_xgb,
                            fit_params_xgb
                            )
    base_cross_val(bag, X_train, y_train, n_jobs=1)
    '''

    res = []
    y = y_train
    num_train = y_train.shape[0]
    for data in list(combinations(datasets, 2)) + datasets:
        print(data)

        if isinstance(data, str):
            X = joblib.load(data)
        else:
            x0 = joblib.load(data[0])
            x1 = joblib.load(data[1])
            X = pd.concat((x0, x1), axis=1)
            del x0, x1

        X_train = X[:num_train]
        X_test = X[num_train:]
        del X

        # X_train, X_val, y_train, y_val = train_test_split(X,
        #                                                  y,
        #                                                  test_size=0.1,
        #                                                  random_state=42)

        xgbr.fit(X_train,
                 y_train,
                 # eval_metric='rmse',
                 # early_stopping_rounds=30,
                 verbose=True,
                 # eval_set=[(X_val, y_val)],
                 )

        y_pred = xgbr.predict(X_test)

        res.append(y_pred)

    final_res = np.mean(res, axis=0)
    final_res[final_res < 1] = 1
    final_res[final_res > 3] = 3
    pd.DataFrame({"id": id_test, "relevance": final_res}).to_csv('xgbr_sub_20160423_bag.csv',index=False)

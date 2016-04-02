# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:18:03 2016

@author: ur57
"""

from sklearn.cross_validation import cross_val_score, KFold
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import shuffle

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords
import re

import numpy as np
import time

from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from sklearn.svm import SVR, LinearSVR

from sklearn.preprocessing import normalize

from scipy.sparse import hstack
from scipy.optimize import minimize

import random
from numpy import random as np_random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.constraints import nonneg
from keras.layers.advanced_activations import PReLU
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

import theano

theano.config.openmp = True

random.seed(2016)
np_random.seed(2016)

stop = stopwords.words('english')

wnl = WordNetLemmatizer()
snow = SnowballStemmer('english')
porter = PorterStemmer()

N_JOBS = 20

class TimeCount(object):
    def __init__(self):
        self.start_time = time.time()

    def done(self, msg):
        print("%s: %s minutes ---" % (msg, round(((time.time() - self.start_time)/60), 2)))
        self.start_time = time.time()

def load_data(samples=None):
    timer = TimeCount()

    if not samples == None:
        df_train = pd.read_csv('train.csv', encoding='ISO-8859-1')[:samples]
        df_test = pd.read_csv('test.csv', encoding='ISO-8859-1')[:samples]
    else:
        df_train = pd.read_csv('train.csv', encoding='ISO-8859-1')
        df_test = pd.read_csv('test.csv', encoding='ISO-8859-1')

    df_desc = pd.read_csv('product_descriptions.csv')

    df_attr = pd.read_csv('attributes.csv')
    df_attr.dropna(inplace=True)

    # Pegue todos os materiais, brands e funcões (usadas pelos avaliadores)
    df_brand = df_attr[df_attr['name'] == 'MFG Brand Name'][['product_uid', 'value']]
    df_brand['brand'] = df_brand['value']
    df_brand.drop('value', axis=1, inplace=True)


    material = dict()
    df_attr['about_material'] = df_attr['name'].str.lower().str.contains('material')
    for row in df_attr[df_attr['about_material']].iterrows():
        r = row[1]
        product = r['product_uid']
        value = r['value']
        material.setdefault(product, '')
        material[product] = material[product] + ' ' + str(value)
    df_material = pd.DataFrame.from_dict(material, orient='index')
    df_material = df_material.reset_index()
    df_material.columns = ['product_uid', 'material']

    color = dict()
    df_attr['about_color'] = df_attr['name'].str.lower().str.contains('color')
    for row in df_attr[df_attr['about_color']].iterrows():
        r = row[1]
        product = r['product_uid']
        value = r['value']
        color.setdefault(product, '')
        color[product] = color[product] + ' ' + str(value)
    df_color = pd.DataFrame.from_dict(color, orient='index')
    df_color = df_color.reset_index()
    df_color.columns = ['product_uid', 'color']

    timer.done("Carregando dados")

    return (df_train, df_brand, df_material, df_color, df_desc, df_test)


def second_process(df_train, df_brand, df_material, df_color, df_desc, df_test):
    timer = TimeCount()

    num_train = df_train.shape[0]
    id_test = df_test['id']
    y = df_train['relevance'].values

    if df_test is not None:
        df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
        del df_test
    else:
        df_all = df_train

    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_color, how='left', on='product_uid')
    #print(df_all.shape)

    del df_brand, df_material, df_desc, df_color

    df_all.fillna(0, inplace=True)

    columns = ['search_term', 'product_title', 'product_description',
                'brand', 'material', 'color']

    timer.done("Finalizando primeiro processamento")

    tfidf = TfidfVectorizer(ngram_range=(2, 5), analyzer='char_wb', strip_accents='unicode')
    #tfidf = CountVectorizer(ngram_range=(2, 5), analyzer='char', strip_accents='unicode')
    tsvd = TruncatedSVD(n_components=500)
    x = None
    for col in columns:
        df_all[col][df_all[col] == np.nan] = ''
        res = tfidf.fit_transform(df_all[col])

        if x is None:
            x = res
        else:
            x = hstack([x, res])
        print(x.shape)
        timer.done('TFIDF for column {}'.format(col))

    x = tsvd.fit_transform(x)
    print(x.shape)
    timer.done("Fim do SVD")

    x_train = x[:num_train]
    x_test = x[num_train:]
    return (x_train, y, x_test, id_test)


def process_data(df_train, df_brand, df_material, df_color, df_desc, df_test):
    timer = TimeCount()

    num_train = df_train.shape[0]
    id_test = df_test['id']
    y = df_train['relevance'].values

    def str_stemmer(s):
        s = str(s)
        s = s.lower()
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) ##'desgruda' palavras que estão juntas

        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)

        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)

        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)

        s = s.replace(" x ", " xby ")
        s = s.replace("*", " xby ")
        s = s.replace(" by ", " xby")
        s = s.replace("x0", " xby 0")
        s = s.replace("x1", " xby 1")
        s = s.replace("x2", " xby 2")
        s = s.replace("x3", " xby 3")
        s = s.replace("x4", " xby 4")
        s = s.replace("x5", " xby 5")
        s = s.replace("x6", " xby 6")
        s = s.replace("x7", " xby 7")
        s = s.replace("x8", " xby 8")
        s = s.replace("x9", " xby 9")
        s = s.replace("0x", "0 xby ")
        s = s.replace("1x", "1 xby ")
        s = s.replace("2x", "2 xby ")
        s = s.replace("3x", "3 xby ")
        s = s.replace("4x", "4 xby ")
        s = s.replace("5x", "5 xby ")
        s = s.replace("6x", "6 xby ")
        s = s.replace("7x", "7 xby ")
        s = s.replace("8x", "8 xby ")
        s = s.replace("9x", "9 xby ")

        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)

        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)

        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)

        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)

        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)

        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)

        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)

        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)

        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)

        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")

        s = s.replace("  "," ")

        s = " ".join([wnl.lemmatize(word) for word in word_tokenize(s.lower())
                        if word not in stop])

        return s

    def str_common_word(str1, str2):
        '''Return how many times the words in str1 appeared in str2
        '''
        words1 = str1.split()
        words2 = str2.split()
        return sum(words2.count(word) for word in words1)

    def str_common_grams(str1, str2, length=3):
        '''Return how many times the ngrams (of length min_len to max_len) of str1
        appeared on str2
        '''
        grams1 = list(ngrams(str1, length))
        grams2 = list(ngrams(str2, length))
        return sum(grams2.count(gram) for gram in grams1)

    if df_test is not None:
        df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
        del df_test
    else:
        df_all = df_train

    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_color, how='left', on='product_uid')
    #print(df_all.shape)

    del df_brand, df_material, df_desc, df_color

    df_all.fillna('', inplace=True)

    columns = ['search_term', 'product_title', 'product_description',
                'brand', 'material', 'color']

    timer.done("Finalizando primeiro processamento")

    for col in columns:
        df_all[col] = df_all[col].map(lambda x: str_stemmer(x))
        df_all['n_word_'+col] = df_all[col].str.count('\ +')
        df_all['n_char_'+col] = df_all[col].str.count('')
        df_all['2grams_'+col] = df_all[col].apply(lambda x: len(list(ngrams(x, 2))))
        df_all['3grams_'+col] = df_all[col].apply(lambda x: len(list(ngrams(x, 3))))
        df_all['4grams_'+col] = df_all[col].apply(lambda x: len(list(ngrams(x, 4))))

        if not col == 'search_term':
            df_all['n_search_word_in_' + col] = df_all.apply(lambda x: str_common_word(x['search_term'], x[col]), axis=1)
            df_all['word_ratio_' + col] = (df_all['n_search_word_in_' + col] / df_all['n_word_search_term'])

            df_all['n_search_2grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 2), axis=1)
            df_all['n_search_3grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 3), axis=1)
            df_all['n_search_4grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 4), axis=1)

            df_all['2grams_raio_' +  col] = (df_all['n_search_2grams_in_'+col] / df_all['2grams_search_term'])
            df_all['3grams_raio_' +  col] = (df_all['n_search_3grams_in_'+col] / df_all['3grams_search_term'])
            df_all['4grams_raio_' +  col] = (df_all['n_search_4grams_in_'+col] / df_all['4grams_search_term'])
        timer.done("Finalizado coluna "  + col + str(df_all.shape))

    df_brand = pd.unique(df_all.brand.ravel())
    d = {}
    i = 1
    for s in df_brand:
        d[s] = i
        i += 1
    df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])

    x = df_all.drop(['id', 'product_uid', 'relevance'] + columns, axis=1).values
    x = np.nan_to_num(x.astype('float32'))

    tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
    tsvd = TruncatedSVD(n_components=15, algorithm='arpack')
    for col in columns:
        tf = tfidf.fit_transform(df_all[col])
        x = np.concatenate((x,
                            tsvd.fit_transform(tf),
                            ), axis=1)
        x = np.concatenate((x, ), axis=1)

    timer.done("Fim do tfidf")

    x_train = x[:num_train]
    x_test = x[num_train:]
    return (x_train, y, x_test, id_test)


def rmse(y, y_pred):
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y, y_pred)
    return mse ** 0.5


def base_cross_val(est, x, y, fit_params=None):
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    res = cross_val_score(est,
                          x,
                          y,
                          cv=3,
                          scoring=rmse_scorer,
                          verbose=3,
                          n_jobs=1,
                          fit_params=fit_params,
                          )
    print(np.mean(res), np.std(res))


def base_randomized_grid_search(est, x, y, params, fit_params=None):
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    grid = RandomizedSearchCV(est,
                              params,
                              verbose=0,
                              scoring=rmse_scorer,
                              error_score=-100,
                              n_jobs=N_JOBS,
                              fit_params=fit_params,
                              cv=3,
                              n_iter=30,
                              )

    grid.fit(x, y)
    return grid


def base_grid_search(est, x, y, params, fit_params=None):
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    grid = GridSearchCV(est,
                        params,
                        verbose=3,
                        scoring=rmse_scorer,
                        error_score=-100,
                        n_jobs=N_JOBS,
                        fit_params=fit_params,
                        cv=3
                        )

    grid.fit(x, y)
    return grid


def gbr_grid_search(x, y, random=False):
    gbr = GradientBoostingRegressor()
    params_gbr = {'loss': ['ls'],   # , 'lad', 'huber', 'quantile'],
                  'learning_rate': np.linspace(0.001, 0.1, 5),
                  'max_depth': np.linspace(3, 15, 5, dtype='int'),
                  'n_estimators': np.linspace(2000, 5000, 5, dtype='int'),
                  'subsample': np.linspace(0.4, 1.0, 5),
                  'max_features': ['sqrt']  # , 'log2', None]
                  }

    if not random:
        grid = base_grid_search(gbr, x, y, params_gbr)
    else:
        grid = base_randomized_grid_search(gbr, x, y, params_gbr)
    return grid


def rfr_grid_search(x, y, random=False):
    rf = RandomForestRegressor()
    params_rf = {'oob_score': [True, False],
                 # 'bootstrap': [True, False],
                 'max_features': ['sqrt', 'log2', None],
                 'max_depth': [10, 15, 20],
                 'n_estimators': np.linspace(1000, 5000, 5, dtype='int'),
                 }

    if not random:
        grid = base_grid_search(rf, x, y, params_rf)
    else:
        grid = base_randomized_grid_search(rf, x, y, params_rf)

    return grid


def xgbr_grid_search(x, y, random=False):
    xgbr = XGBRegressor(nthread=1)
    params_xgb = {'max_depth': np.linspace(7, 15, 5, dtype='int32'),
                  'learning_rate': np.linspace(0.001, 0.1, 5),
                  'subsample': np.linspace(0.3, 1, 5),
                  'colsample_bytree': np.linspace(0.1, 0.7, 5),
                  'n_estimators': np.linspace(3000, 5000, 5, dtype='int'),
                  }

    fit_params_xgb = {'eval_metric': 'rmse'}


    if not random:
        grid = base_grid_search(xgbr, x, y, params_xgb, fit_params_xgb)
    else:
        grid = base_randomized_grid_search(xgbr, x, y, params_xgb, fit_params_xgb)

    return grid


def bagr_grid_search(x, y, base=RandomForestRegressor(), random=False):
    bagr = BaggingRegressor(base)
    params_bagr = {'max_samples': np.linspace(0.5, 1, 5),
                   'n_estimators': np.linspace(500, 1000, 5, dtype='int'),
                   'max_features': np.linspace(0.3, 1, 5),
                   }
    if not random:
        grid = base_grid_search(bagr, x, y, params_bagr)
    else:
        grid = base_randomized_grid_search(bagr, x, y, params_bagr)

    return grid


def svr_rbf_grid_search(x, y, random=False):
    svr = SVR()
    params_svr = {'C': np.linspace(0.01, 1.0, 5),
                  'epsilon': np.linspace(0.0, 1.0, 5),
                  'shrinking': [True, False],
                  'tol': np.linspace(0.0001, 0.001, 5),
                  'kernel': ['rbf'],
                  }
    if not random:
        grid = base_grid_search(svr, x, y, params_svr)
    else:
        grid = base_randomized_grid_search(svr, x, y, params_svr)

    return grid


def svr_poly_grid_search(x, y, random=False):
    svr = SVR()
    params_svr = {'C': np.linspace(0.01, 1.0, 50),
                  'epsilon': np.linspace(0.0, 1.0, 5),
                  'degree': np.linspace(3, 10, 5, dtype='int'),
                  'shrinking': [True, False],
                  'tol': np.linspace(0.0001, 0.001, 5),
                  'kernel': ['poly'],
                  }
    if not random:
        grid = base_grid_search(svr, x, y, params_svr)
    else:
        grid = base_randomized_grid_search(svr, x, y, params_svr)

    return grid


def svr_sigmoid_grid_search(x, y, random=False):
    svr = SVR()
    params_svr = {'C': np.linspace(0.01, 1.0, 5),
                  'epsilon': np.linspace(0.0, 1.0, 5),
                  'shrinking': [True, False],
                  'tol': np.linspace(0.0001, 0.001, 5),
                  'kernel': ['sigmoid'],
                  }
    if not random:
        grid = base_grid_search(svr, x, y, params_svr)
    else:
        grid = base_randomized_grid_search(svr, x, y, params_svr)

    return grid


def svr_linear_grid_search(x, y, random=False):
    svr = LinearSVR()
    params_svr = {'C': np.linspace(0.01, 1.0, 5),
                  'epsilon': np.linspace(0.0, 1.0, 5),
                  'tol': np.linspace(0.0001, 0.001, 5),
                  }
    if not random:
        grid = base_grid_search(svr, x, y, params_svr)
    else:
        grid = base_randomized_grid_search(svr, x, y, params_svr)

    return grid


def get_keras(n_columns=356, optimizer='sgd'):
    def rmse(y_true, y_pred):
        from keras import backend as k
        from keras.objectives import mean_squared_error

        return k.sqrt(mean_squared_error(y_true, y_pred))

    model = Sequential()
    #model.add(BatchNormalization()
    model.add(Dense(512, input_dim=n_columns))
    model.add(LeakyReLU())
    model.add(Dense(512))
    model.add(LeakyReLU())

    model.add(Dense(512))
    model.add(LeakyReLU())
    model.add(Dense(512))
    model.add(LeakyReLU())
    model.add(Dense(1, activation='relu'))

    model.compile(loss=rmse, optimizer=optimizer)
    return model

def learn_reducer(epoch):
    return 0.3 / (3 *(epoch + 1))

def do_keras(X_train, X_test, y_train, y_test):
    model = get_keras(X_train.shape[1])
    model.fit(X_train, y_train, batch_size=64)
    return model.evaluate(X_test, y_test)

def optimize_function(pesos, preds, y_true):
    weigths = np.asarray(pesos)
    values = np.asarray(preds)
    y_pred = values * np.expand_dims(weigths, 1)

    y_pred = np.sum(y_pred, axis=0)/np.sum(weigths)
    y_pred[y_pred > 3] = 3
    y_pred[y_pred < 1] = 1
    res = rmse(y_true, y_pred)
    #print('Resultado para {} = {}'.format(weigths, res))
    return res


class Stacker(object):
    def __init__(self, stacker, base_models, folds=4):
        self.folds = folds
        self.base_models = sorted(base_models)
        self.stacker = stacker

    def fit(self, X, y):
        folds = list(KFold(len(y), n_folds=self.folds, random_state=0))
        base_preds = [self._mini_train_predictions(m, X, y, folds) for m in self.base_models]
        n = len(y)

        XX = np.hstack([X] + [p.reshape(n, 1) for p in base_preds])
        self.stacker.fit(XX, y)
        for m in self.base_models:
            m.fit(X, y)
        return self

    def predict(self, X):
        n = X.shape[0]
        XX = np.hstack([X] + [m.predict(X).reshape(n, 1) for m in self.base_models])
        return self.stacker.predict(XX)

    def _mini_train_predictions(self, model, X, y, train_test_folds):
        ind2pred = {}
        for i, (train, test) in enumerate(train_test_folds):
            model.fit(X[train], y[train])
            preds = model.predict(X[test])
            for i, p in zip(test, preds):
                ind2pred[i] = p

        return np.array([ind2pred[i] for i in range(len(y))])

class MetaRegressor(BaseEstimator):
    def fit(self, x, y=None, **fit_params):
        timer = TimeCount()
        grids = []
        self.scores = []
        self.estimators = []


        grids.append(xgbr_grid_search(x, y, True))
        timer.done("XGBR")

        grids.append(gbr_grid_search(x, y, True))
        timer.done("GBR")

        grids.append(rfr_grid_search(x, y, True))
        timer.done("RFR")

        grids.append(bagr_grid_search(x, y, random=True))
        timer.done("BAGR")

        '''
        grids.append(svr_rbf_grid_search(x, y, random=True))
        timer.done("SVR - RBF")

        grids.append(svr_poly_grid_search(x, y, random=True))
        timer.done("SVR - POLY")

        grids.append(svr_sigmoid_grid_search(x, y, random=True))
        timer.done("SVR - Sigmoid")

        grids.append(svr_linear_grid_search(x, y, random=True))
        timer.done("SVR - Linear")
        '''


        for grid in grids:
            self.scores.append(grid.best_score_ * -1)
            est = grid.best_estimator_
            self.estimators.append(est)
            print("{} ({}) = {} ".format(est.__class__,
                                         grid.best_score_,
                                         grid.best_params_))


        '''
        keras = KerasRegressor(get_keras,
                               optimizer='adadelta',
                               n_columns=x.shape[1],
                               batch_size=5,
                               nb_epoch=100,
                               validation_split=0.1,
                               #shuffle=True,
                               callbacks=[
                                          #EarlyStopping(patience=3, mode='min'),
                                          #LearningRateScheduler(learn_reducer)
                                          ])
        keras.fit(x, y)
        self.estimators.append(keras)
        pred = keras.predict(x)
        print('{:.4f} +/- {:.4f}'.format(np.mean(pred),np.std(pred)))


        keras = KerasRegressor(get_keras,
                               optimizer='adam',
                               n_columns=x.shape[1],
                               batch_size=5,
                               nb_epoch=100,
                               validation_split=0.1,
                               #shuffle=True,
                               callbacks=[
                                          #EarlyStopping(patience=3, mode='min'),
                                          #LearningRateScheduler(learn_reducer)
                                          ])
        keras.fit(x, y)
        self.estimators.append(keras)
        pred = keras.predict(x)
        print('{:.4f} +/- {:.4f}'.format(np.mean(pred),np.std(pred)))


        keras = KerasRegressor(get_keras,
                               optimizer='adamax',
                               n_columns=x.shape[1],
                               batch_size=5,
                               nb_epoch=100,
                               validation_split=0.1,
                               #shuffle=True,
                               callbacks=[
                                          #EarlyStopping(patience=3, mode='min'),
                                          #LearningRateScheduler(learn_reducer)
                               ])
        keras.fit(x, y)
        self.estimators.append(keras)
        pred = keras.predict(x)
        print('{:.4f} +/- {:.4f}'.format(np.mean(pred),np.std(pred)))
        '''
        '''
        self.estimators.append(XGBRegressor(n_estimators=5000, nthread=8).fit(x,y))
        self.estimators.append(GradientBoostingRegressor(n_estimators=5000).fit(x,y))
        self.estimators.append(RandomForestRegressor(n_estimators=2500, n_jobs=8).fit(x,y))
        self.estimators.append(BaggingRegressor(n_estimators=1000, n_jobs=8).fit(x,y))
        '''
        timer.done("Estimacoes iniciais")


        return self

    def predict(self, x):

        preds = []
        for est in self.estimators:
            pred = est.predict(x)
            pred = np.reshape(pred, (pred.shape[0],))
            preds.append(pred)

        preds = np.asarray(preds)

        self.scores = np.asarray(self.scores) * 1
        factor = np.min(self.scores) / self.scores
        factor = factor ** 1

        preds = np.expand_dims(factor, 1) * preds

        y_pred = np.sum(preds, axis=0)/np.sum(factor)


        y_pred[y_pred > 3] = 3
        y_pred[y_pred < 1] = 1

        return y_pred


if __name__ == '__main__':
    #x, y, x_test, id_test = process_data(*load_data())
    # x, y, x_test, id_test = second_process(*load_data(5000))
    #joblib.dump(x, 'x.pkl')
    #joblib.dump(y, 'y.pkl')
    #joblib.dump(x_test, 'x_test.pkl')
    #joblib.dump(id_test, 'id_test.pkl')
    x = joblib.load('x.pkl')[:20000]
    y = joblib.load('y.pkl')[:20000]
    x_test = joblib.load('x_test.pkl')
    id_test = joblib.load('id_test.pkl')

    xbin1 = np.repeat(x[(y >= 1.0) & (y < 1.5)], 7, axis=0)
    xbin2 = np.repeat(x[(y >= 1.5) & (y < 2.0)], 5, axis=0)
    xbin3 = np.repeat(x[(y >= 2.0) & (y < 2.5)], 1, axis=0)
    xbin4 = np.repeat(x[(y >= 2.5) & (y <= 3)], 1, axis=0)

    x = np.vstack((xbin1, xbin2, xbin3, xbin4))

    ybin1 = np.repeat(y[(y >= 1.0) & (y < 1.5)], 7, axis=0)
    ybin2 = np.repeat(y[(y >= 1.5) & (y < 2.0)], 5, axis=0)
    ybin3 = np.repeat(y[(y >= 2.0) & (y < 2.5)], 1, axis=0)
    ybin4 = np.repeat(y[(y >= 2.5) & (y <= 3)], 1, axis=0)

    y = np.concatenate((ybin1, ybin2, ybin3, ybin4))

    x, y = shuffle(x, y)

    x = normalize(x)
    x_test = normalize(x_test)
    est = MetaRegressor()

    #print(do_keras(*train_test_split(x, y, test_size=0.25)))


    base_cross_val(est, x, y)
    base_cross_val(XGBRegressor(n_estimators=500), x, y)
    '''
    est.fit(x, y)
    y_pred = est.predict(x_test)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('new_meta_submission.csv',index=False)
    '''



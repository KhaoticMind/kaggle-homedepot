# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:18:03 2016

@author: ur57
"""

from sklearn.cross_validation import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import everygrams

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

import random
random.seed(2016)

stop = stopwords.words('english')

wnl = WordNetLemmatizer()
snow = SnowballStemmer('english')
porter = PorterStemmer()



class TimeCount(object):
    def __init__(self):
        self.start_time = time.time()

    def done(self, msg):
        print("%s: %s minutes ---" % (msg, round(((time.time() - self.start_time)/60), 2)))
        self.start_time = time.time()


class xgbDepot(XGBRegressor):

    def predict(self, data, output_margin=False, ntree_limit=0):
        y_pred = super(XGBRegressor,self).predict(data, output_margin, ntree_limit)
        y_pred[y_pred > 3] = 3
        y_pred[y_pred < 1] = 1

        '''
        # Normalizar os resutlados dentro da faixa esperada
        # Durante o cross validation so piora os dados
        res = []
        foo = [1.00, 1.25, 1.33, 1.50, 1.67, 1.75, 2.00, 2.25, 2.33, 2.50, 2.67, 2.75, 3.00]
        for i in range(len(foo) - 1):
            res.append( (foo[i], foo[i+1]))

        for inicio, fim in res:
            media = (inicio + fim) / 2
            y_pred[(y_pred > inicio) & (y_pred < media)] = inicio
            y_pred[(y_pred > media)  & (y_pred < fim)] = fim
        '''

        return y_pred

def load_data():
    timer = TimeCount()

    df_train = pd.read_csv('train.csv', encoding='ISO-8859-1')

    df_desc = pd.read_csv('product_descriptions.csv')

    df_attr = pd.read_csv('attributes.csv')
    df_attr.dropna(inplace=True)

    # Pegue todos os materiais, brands e funcões (usadas pelos avaliadores)
    df_brand = df_attr[df_attr['name'] == 'MFG Brand Name'][['product_uid', 'value']]
    df_brand['brand'] = df_brand['value']
    df_brand.drop('value', axis=1, inplace=True)

    df_material = df_attr[df_attr['name'] == 'Material'][['product_uid', 'value']]
    df_material['material'] = df_material['value']
    df_material.drop('value', axis=1, inplace=True)

    df_test = pd.read_csv('test.csv', encoding='ISO-8859-1')

    timer.done("Carregando dados")

    return (df_train, df_brand, df_material, df_desc, df_test)


def process_data(df_train, df_brand, df_material, df_desc, df_test):
    timer = TimeCount()

    num_train = df_train.shape[0]
    id_test = df_test['id']
    y = df_train['relevance'].values

    def str_stemmer(s):
        s = s.replace("  ", " ")
        s = s.replace(",", "") #could be number / segment later
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("-", " ")
        s = s.replace("//", "/")
        s = s.replace("..", ".")
        s = s.replace(" / ", " ")
        s = s.replace(" \\ ", " ")
        s = s.replace(".", " . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x ", " xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xbi ")
        s = s.replace(" by ", " xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  ", " ")
        s = s.replace(" . ", " ")

        s = " ".join([wnl.lemmatize(word) for word in word_tokenize(s.lower())
                        if word not in stop])

        s = s.replace("toliet", "toilet")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("vinal", "vinyl")
        s = s.replace("vynal", "vinyl")
        s = s.replace("skill", "skil")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("rustoleum", "rust-oleum")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")

        return s

    def str_common_word(str1, str2):
        '''Return how many times the words in str1 appeared in str2
        '''
        words1 = str1.split()
        words2 = str2.split()
        return sum(words2.count(word) for word in words1)

    def str_common_grams(str1, str2, min_len=3, max_len=4):
        '''Return how many times the ngrams (of length min_len to max_len) of str1
        appeared on str2
        '''
        grams1 = list(everygrams(str1, min_len, max_len))
        grams2 = list(everygrams(str2, min_len, max_len))
        return sum(grams2.count(gram) for gram in grams1)

    if df_test is not None:
        df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
        del df_test
    else:
        df_all = df_train

    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')
    #df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
    #print(df_all.shape)

    del df_brand, df_material, df_desc

    df_all.fillna('', inplace=True)

    columns = ['search_term', 'product_title', 'product_description',
                'brand'] #'material',

    timer.done("Finalizando primeiro processamento")

    for col in columns:
        df_all[col] = df_all[col].map(lambda x: str_stemmer(x))
        df_all['n_word_'+col] = df_all[col].str.count('\ +')
        #df_all['n_char_'+col] = df_all[col].str.count('')

        if not col == 'search_term':
            df_all['n_search_word_in_' + col] = df_all.apply(lambda x: str_common_word(x['search_term'], x[col]), axis=1)
            df_all['word_ratio_' + col] = (df_all['n_search_word_in_' + col] / df_all['n_word_search_term'])

            df_all['n_search_2grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 2, 2), axis=1)
            df_all['n_search_3grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 3, 3), axis=1)
            df_all['n_search_4grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 4, 4), axis=1)
        timer.done("Finalizado coluna "  + col + str(df_all.shape))

    x = df_all.drop(['id', 'product_uid', 'relevance'] + columns, axis=1).values
    x = np.nan_to_num(x.astype('float32'))

    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state = 2016)
    for col in columns:
        x = np.concatenate((x, tsvd.fit_transform(tfidf.fit_transform(df_all[col]))), axis=1)

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
                          cv=10,
                          scoring=rmse_scorer,
                          verbose=0,
                          n_jobs=1,
                          fit_params=fit_params,
                          )
    print(np.mean(res), np.std(res))


def base_randomized_grid_search(est, x, y, params, fit_params=None):
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    grid = RandomizedSearchCV(est,
                              params,
                              verbose=3,
                              scoring=rmse_scorer,
                              error_score=100,
                              n_jobs=15,
                              fit_params=fit_params,
                              cv=5,
                              n_iter=40,
                              )

    grid.fit(x, y)
    return grid


def base_grid_search(est, x, y, params, fit_params=None):
    rmse_scorer = make_scorer(rmse, greater_is_better=False)
    grid = GridSearchCV(est,
                        params,
                        verbose=3,
                        scoring=rmse_scorer,
                        error_score=100,
                        n_jobs=15,
                        fit_params=fit_params,
                        cv=5
                        )

    grid.fit(x, y)
    return grid


def gbr_grid_search(x, y, random=False):
    gbr = GradientBoostingRegressor(random_state=0, init=rf)
    params_gbr = {'loss': ['ls'],   # , 'lad', 'huber', 'quantile'],
                  'learning_rate': np.linspace(0.0001, 0.5, 10),
                  'max_depth': [10, 15, 20],
                  'n_estimators': [300],
                  'subsample': np.linspace(0.1, 1.0, 10),
                  'max_features': ['sqrt']  # , 'log2', None]
                  }

    if not random:
        grid = base_grid_search(gbr, x, y, params_gbr)
    else:
        grid = base_randomized_grid_search(gbr, x, y, params_gbr)
    return grid


def rfr_grid_search(x, y, random=False):
    rf = RandomForestRegressor(random_state=0)
    params_rf = {'oob_score': [True, False],
                 'bootstrap': [True, False],
                 'max_features': ['sqrt', 'log2', None],
                 'max_depth': [3, 6, 10, 15],
                 'n_estimators': [20, 50, 100, 200]}

    if not random:
        grid = base_grid_search(rf, x, y, params_rf)
    else:
        grid = base_randomized_grid_search(rf, x, y, params_rf)

    return grid


def xgbr_grid_search(x, y, random=False):
    xgbr = XGBRegressor(nthread=1)
    params_xgb = {'max_depth': np.linspace(7, 15, 5, dtype='int32'),
                  'learning_rate': np.linspace(0.01, 0.5, 5),
                  'subsample': np.linspace(0.5, 1, 5),
                  'colsample_bytree': np.linspace(0.5, 1, 5),
                  }

    fit_params_xgb = {'eval_metric': 'rmse'}

    if not random:
        grid = base_grid_search(xgbr, x, y, params_xgb, fit_params_xgb)
    else:
        grid = base_randomized_grid_search(xgbr, x, y, params_xgb, fit_params_xgb)

    return grid


class MetaRegressor(BaseEstimator):
    def fit(self, x, y=None, **fit_params):
        self.xbr = xgbr_grid_search(x, y, True)
        self.gbr = gbr_grid_search(x, y, True)
        self.rfr = rfr_grid_search(x, y, True)

        self.bagr = BaggingRegressor(self.rfr,
                                     n_estimators=100,
                                     max_samples=0.1,
                                     random_state=25)

        self.bagr.fit(x, y)

    def predict(self, hd_searches):
        y_pred_xbr = self.xbr.predict(x_test)
        y_pred_gbr = self.gbr.predict(x_test)
        y_pred_rfr = self.rfr.predict(x_test)
        y_pred_bag = self.bagr.predict(x_test)

        y_pred = (y_pred_xbr + y_pred_gbr + y_pred_rfr + y_pred_bag) / 5

        y_pred[y_pred > 3] = 3
        y_pred[y_pred < 1] = 1

        return y_pred


if __name__ == '__main__':
    # x, y, x_test, id_test = process_data(*load_data())
    # joblib.dump(x, 'x.pkl')
    # joblib.dump(y, 'y.pkl')
    # joblib.dump(x_test, 'x_test.pkl')
    # joblib.dump(id_test, 'id_test.pkl')
    x = joblib.load('x.pkl')
    y = joblib.load('y.pkl')
    x_test = joblib.load('x_test.pkl')
    id_test = joblib.load('id_test.pkl')

    est = MetaRegressor()
    base_cross_val(est, x, y)

    #pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

# rfr
# mean: 0.48109, std: 0.01171, params: {'oob_score': False, 'n_estimators': 200, 'max_depth': 15, 'max_features': 'sqrt', 'bootstrap': False}
# gbr
# mean: 0.48098, std: 0.01196, params: {'loss': 'ls', 'max_depth': 10, 'n_estimators': 200, 'learning_rate': 0.025750000000000002, 'subsample': 0.90000000000000002, 'max_features': 'sqrt'}

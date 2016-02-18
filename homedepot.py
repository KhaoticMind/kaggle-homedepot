# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:18:03 2016

@author: ur57
"""

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

import numpy as np
import time

class TimeCount(object):
    def __init__(self):
        self.start_time = time.time()

    def done(self, msg):
        print("%s: %s minutes ---" % (msg, round(((time.time() - self.start_time)/60), 2)))
        self.start_time = time.time()

def load_data():
    import pandas as pd
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

    # df_test = pd.read_csv('test.csv', encoding='ISO-8859-1')
    df_test = None
    timer.done("Carregando dados")

    return (df_train, df_brand, df_material, df_desc, df_test)


def process_data(df_train, df_brand, df_material, df_desc, df_test):
    timer = TimeCount()
    import pandas as pd
    from nltk.tokenize import word_tokenize
    from nltk.util import everygrams

    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    wnl = WordNetLemmatizer()
    snow = SnowballStemmer('english')

    def stem(x):
        #return wnl.lemmatize(x)
        return snow.stem(x)

    def str_stemmer(s):
        from nltk.corpus import stopwords
        import re
        stop = stopwords.words('english')

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
        s = s.replace("  "," ")
        s = s.replace(" . "," ")

        s = " ".join([stem(word) for word in word_tokenize(s.lower())
                        if word not in stop])

        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")

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

    df_all = df_train
    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

    del df_brand, df_material, df_desc

    df_all.fillna('', inplace=True)

    columns = ['search_term', 'product_title', 'product_description',
               'material', 'brand']

    timer.done("Finalizando primeiro processamento")
    for col in columns:
        df_all[col] = df_all[col].map(lambda x: str_stemmer(x))
        df_all['n_word_'+col] = df_all[col].str.count('\ +')
        df_all['n_char_'+col] = df_all[col].str.count('')
        if not col == 'search_term':
            df_all['n_search_word_in_' + col] = df_all.apply(lambda x: str_common_word(x['search_term'], x[col]), axis=1)
            df_all['word_ratio_' + col] = (df_all['n_search_word_in_' + col] / df_all['n_word_search_term'])

            df_all['n_search_2grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 2, 2), axis=1)
            df_all['n_search_3grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 3, 3), axis=1)
            df_all['n_search_4grams_in_'+ col] = df_all.apply(lambda x: str_common_grams(x['search_term'], x[col], 4, 4), axis=1)
        timer.done("Finalizado coluna " + col)

    y = df_all['relevance'].values
    x = df_all.drop(['id', 'product_uid', 'relevance'] + columns, axis=1).values
    x = np.nan_to_num(x.astype('float32'))

    return (x, y)


def rmse(y, y_pred):
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y, y_pred)
    return mse ** 0.5

if __name__ == '__main__':

    from sklearn.externals import joblib
    from sklearn.metrics import make_scorer
    from xgboost import XGBRegressor
    import sys

    x, y = process_data(*load_data())
    joblib.dump(x, 'x.pkl')
    joblib.dump(y, 'y.pkl')
    # x = joblib.load('x.pkl')
    # y = joblib.load('y.pkl')

    rmse_scorer = make_scorer(rmse, greater_is_better=False)


    xgbr = XGBRegressor(nthread=15,
                        n_estimators=2000,
                        colsample_bytree=0.7,
                        min_child_weight=4.0,
                        subsample=0.55,
                        learning_rate=0.05,
                        #gamma=0.75
                        )
    #xgbr.fit(x, y, eval_metric='rmse')

    rmse = cross_val_score(xgbr,
                           x,
                           y,
                           cv=5,
                           scoring=rmse_scorer,
                           verbose=255,
                           n_jobs=1,
                           fit_params={'eval_metric':'rmse'}
                           )
    print(np.mean(rmse), np.std(rmse))
    sys.exit(0)


    rf = RandomForestRegressor(n_estimators=100,
                               random_state=0)

    rf = RandomForestRegressor(**{'oob_score': False,
                                  'n_estimators': 50,
                                  'max_depth': 15,
                                  'max_features': 'sqrt',
                                  'bootstrap': False,
                                  'random_state' : 0})
    clf = BaggingRegressor(rf,
                           n_estimators=100,
                           max_samples=0.1,
                           random_state=25)

#    rmse = cross_val_score(clf,
#                           x,
#                           y,
#                           cv=5,
#                           scoring=rmse_scorer,
#                           verbose=255,
#                           n_jobs=-1)
#    print(np.mean(rmse), np.std(rmse))


    rf = RandomForestRegressor(random_state=0)
    params_rf = {'oob_score': [True, False],
                 'bootstrap': [True, False],
                 'max_features': ['sqrt', 'log2', None],
                 'max_depth': [3, 6, 10, 15],
                 'n_estimators': [20, 50, 100, 200]}
    grid_rf = GridSearchCV(rf,
                           params_rf,
                           refit=False,
                           verbose=255,
                           scoring=rmse_scorer,
                           error_score=100,
                           n_jobs=15,
                           # n_iter=20
                           )
    # grid_rf.fit(x, y)
    # print(grid_rf.best_score_)
    # print(grid_rf.best_params_)

    gbr = GradientBoostingRegressor(random_state=0, init=rf)
    params_gbr = {'loss': ['ls' ],#, 'lad', 'huber', 'quantile'],
                  'learning_rate': np.linspace(0.0001, 0.5, 10),
                  'max_depth': [10, 15, 20],
                  'n_estimators': [100],
                  'subsample': np.linspace(0.1, 1.0, 10),
                  'max_features': ['sqrt']#, 'log2', None]
                  }

    grid_gbr = RandomizedSearchCV(gbr,
                            params_gbr,
                            refit=False,
                            verbose=255,
                            scoring=rmse_scorer,
                            error_score=-100,
                            cv=5,
                            n_jobs=15,
                            pre_dispatch='n_jobs',
                            n_iter=20)
    grid_gbr.fit(x, y)
    print(grid_gbr.best_score_)
    print(grid_gbr.best_params_)


    xgb_reg = XGBRegressor(nthread=1)
    params_xgb = {'max_depth': np.linspace(3, 15, 5),
                  'learning_rate': np.linspace(0.001, 0.1, 5),
                  'subsample': np.linspace(0.1, 0.9, 5),
                  'colsample_bytree': np.linspace(0.1, 0.9, 5)}
    fit_params_xgb = {'eval_metric': 'rmse'}
    grid_xgb = GridSearchCV(xgb_reg,
                            params_xgb,
                            refit=False,
                            verbose=3,
                            scoring=rmse_scorer,
                            error_score=100,
                            n_jobs=15,
                            fit_params=fit_params_xgb)


# rfr
# mean: 0.48109, std: 0.01171, params: {'oob_score': False, 'n_estimators': 200, 'max_depth': 15, 'max_features': 'sqrt', 'bootstrap': False}
# gbr
# mean: 0.48098, std: 0.01196, params: {'loss': 'ls', 'max_depth': 10, 'n_estimators': 200, 'learning_rate': 0.025750000000000002, 'subsample': 0.90000000000000002, 'max_features': 'sqrt'}
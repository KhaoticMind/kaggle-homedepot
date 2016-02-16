# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:18:03 2016

@author: ur57
"""

import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import everygrams

import numpy as np

from sklearn.metrics import make_scorer

def load_train():
    return pd.read_csv('train.csv', encoding='ISO-8859-1')

def process_data():

if __name__ == '__main__':

    def rmse(y, y_pred):
        from sklearn.metrics import mean_squared_error
        from numpy import sqrt
        mse = mean_squared_error(y, y_pred)
        if mse is not None:
            return sqrt(mse) * -1
        else:
            return None

    rmse_scorer = make_scorer(rmse, greater_is_better=False)

    stop = stopwords.words('english')

    wnl = WordNetLemmatizer()
    stem = SnowballStemmer('english')

    # stem = lambda x: stemmer.stem(x)

    def stem(x):
        return wnl.lemmatize(x)

    def str_stemmer(s):
        return " ".join([stem(word) for word in word_tokenize(s.lower())
                        if word not in stop])

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

    df_train = load_train()
    # df_test = pd.read_csv('test.csv', encoding='ISO-8859-1')

    df_all = df_train

    df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
    df_all = pd.merge(df_all, df_desc, how='left', on='product_uid')

    del df_brand, df_material, df_desc

    df_all.fillna('', inplace=True)

    columns = ['search_term', 'product_title', 'product_description',
               'material', 'brand']

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

    y = df_all['relevance'].values
    x = df_all.drop(['id', 'product_uid', 'relevance'] + columns, axis=1).values
    x = np.nan_to_num(x.astype('float32'))

    del df_all

    rf = RandomForestRegressor(n_estimators=50, random_state=0, verbose=3)
    clf = BaggingRegressor(rf, n_estimators=50, max_samples=0.1, random_state=25, verbose=3)
    #mse = cross_val_score(clf, x, y, cv=5, scoring='mean_squared_error', verbose=3)
    #rmse = (mse * -1) ** 0.5
    #print(np.mean(rmse), np.std(rmse))

    rf = RandomForestRegressor(random_state=0)
    params_rf = {'oob_score': [True, False],
                 'bootstrap': [True, False],
                 'max_features': ['sqrt', 'log2', None],
                 'max_depth': [3, 6, 10, 15],
                 'n_estimators': [20, 50, 100, 200]}
    grid_rf = GridSearchCV(rf,
                           params_rf,
                           refit=False,
                           verbose=3,
                           scoring=rmse_scorer,
                           error_score=100,
                           n_jobs=8)
    grid_rf.fit(x, y)

    gbr = GradientBoostingRegressor(random_state=0)
    params_gbr = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                  'learning_rate': np.linspace(0.001, 0.1, 5),
                  'max_depth': [3, 6, 10, 15],
                  'n_estimators': [20, 50, 100, 200],
                  'subsample': np.linspace(0.1, 0.9, 5),
                  'max_features': ['sqrt', 'log2', None]}

    grid_gbr = GridSearchCV(gbr,
                            params_gbr,
                            refit=False,
                            verbose=3,
                            scoring=rmse_scorer,
                            error_score=100,
                            n_jobs=8)
    grid_gbr.fit(x, y)


# rfr
# mean: 0.48109, std: 0.01171, params: {'oob_score': False, 'n_estimators': 200, 'max_depth': 15, 'max_features': 'sqrt', 'bootstrap': False}
# gbr
# mean: 0.48098, std: 0.01196, params: {'loss': 'ls', 'max_depth': 10, 'n_estimators': 200, 'learning_rate': 0.025750000000000002, 'subsample': 0.90000000000000002, 'max_features': 'sqrt'}
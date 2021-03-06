import time
start_time = time.time()

import numpy as np

import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
from nltk.util import everygrams
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import Normalizer, MinMaxScaler

#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
import re
#import enchant
import random

from homedepot import *
from helper_processing import *
from keras.wrappers.scikit_learn import KerasRegressor

from itertools import product as iter_product

from xgboost import XGBRegressor
random.seed(2016)

df_train = pd.read_csv('train.csv', encoding="ISO-8859-1", nrows=30000)
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1", nrows=200)
df_pro_desc = pd.read_csv('product_descriptions.csv')
df_attr = pd.read_csv('attributes.csv')
df_attr.dropna(inplace=True)

df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})

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

num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
df_all = pd.merge(df_all, df_color, how='left', on='product_uid')

def str_stem(s):
    if isinstance(s, str):
        s = s.lower()
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) ##'desgruda' palavras que estão juntas

        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)

        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)

        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)

        s = s.replace(" x "," xby ")
        s = s.replace("*"," xby ")
        s = s.replace(" by "," xby")
        s = s.replace("x0"," xby 0")
        s = s.replace("x1"," xby 1")
        s = s.replace("x2"," xby 2")
        s = s.replace("x3"," xby 3")
        s = s.replace("x4"," xby 4")
        s = s.replace("x5"," xby 5")
        s = s.replace("x6"," xby 6")
        s = s.replace("x7"," xby 7")
        s = s.replace("x8"," xby 8")
        s = s.replace("x9"," xby 9")
        s = s.replace("0x","0 xby ")
        s = s.replace("1x","1 xby ")
        s = s.replace("2x","2 xby ")
        s = s.replace("3x","3 xby ")
        s = s.replace("4x","4 xby ")
        s = s.replace("5x","5 xby ")
        s = s.replace("6x","6 xby ")
        s = s.replace("7x","7 xby ")
        s = s.replace("8x","8 xby ")
        s = s.replace("9x","9 xby ")

        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)

        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)

        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)

        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)

        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)

        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)

        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)

        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)

        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)

        s = s.replace("  +"," ")
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        return s.lower()
    else:
        return " "

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_common_grams(str1, str2, min_len=3, max_len=4):
        '''Return how many times the ngrams (of length min_len to max_len) of str1
        appeared on str2
        '''
        grams1 = list(everygrams(str1, min_len, max_len))
        grams2 = list(everygrams(str2, min_len, max_len))
        return sum(grams2.count(gram) for gram in grams1)

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None, **fit_params):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','product_title','product_description','product_info','attr','brand', 'material', 'color']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None, **fit_params):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

#comment out the lines below use df_all.csv for further grid search testing
#if adding features consider any drops on the 'cust_regression_vals' class
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
df_all['material'] = df_all['material'].map(lambda x:str_stem(x))
df_all['color'] = df_all['color'].map(lambda x:str_stem(x))
df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']+"\t"+df_all['material']+"\t"+df_all['color']

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_material'] = df_all['material'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_color'] = df_all['color'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']

df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_material'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['word_in_color'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))

df_all['2grams_search_term'] = df_all['search_term'].apply(lambda x: len(list(ngrams(x, 2))))
df_all['3grams_search_term'] = df_all['search_term'].apply(lambda x: len(list(ngrams(x, 3))))
df_all['4grams_search_term'] = df_all['search_term'].apply(lambda x: len(list(ngrams(x, 4))))

df_all['2grams_in_title'] = df_all['product_info'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[1], 2, 2))
df_all['2grams_in_description'] = df_all['product_info'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[2], 2, 2))
df_all['2grams_in_brand'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[1], 2, 2))
df_all['2grams_in_material'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[2], 2, 2))
df_all['2grams_in_color'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[3], 2, 2))

df_all['3grams_in_title'] = df_all['product_info'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[1], 3, 3))
df_all['3grams_in_description'] = df_all['product_info'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[2], 3, 3))
df_all['3grams_in_brand'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[1], 3, 3))
df_all['3grams_in_material'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[2], 3, 3))
df_all['3grams_in_color'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[3], 3, 3))

df_all['4grams_in_title'] = df_all['product_info'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[1], 4, 4))
df_all['4grams_in_description'] = df_all['product_info'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[2], 4, 4))
df_all['4grams_in_brand'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[1], 4, 4))
df_all['4grams_in_material'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[2], 4, 4))
df_all['4grams_in_color'] = df_all['attr'].map(lambda x:str_common_grams(x.split('\t')[0],x.split('\t')[3], 4, 4))

for col in ['title', 'description', 'brand', 'material', 'color']:
    df_all['2gram_ratio_' + col] = df_all['2grams_in_' + col] / df_all['2grams_search_term']
    df_all['3gram_ratio_' + col] = df_all['3grams_in_' + col] / df_all['3grams_search_term']
    df_all['4gram_ratio_' + col] = df_all['4grams_in_' + col] / df_all['4grams_search_term']


df_all['w_ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['w_ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['w_ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
df_all['w_ratio_material'] = df_all['word_in_material']/df_all['len_of_material']
df_all['w_ratio_color'] = df_all['word_in_color']/df_all['len_of_color']

print("--- Primeiro processamento: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()


w2v = Word2Vec.load_word2vec_format('word2vec-GoogleNews-vectors-negative300.bin.gz', binary=True)
print("--- Loading W2V: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()

df_all['w2v_search_desc_1_1'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 1, 1))
df_all['w2v_search_desc_1_2'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 1, 2))
df_all['w2v_search_desc_1_3'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 1, 3))

df_all['w2v_search_desc_2_1'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 2, 1))
df_all['w2v_search_desc_2_2'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 2, 2))
df_all['w2v_search_desc_2_3'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 2, 3))

df_all['w2v_search_desc_3_1'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 3, 1))
df_all['w2v_search_desc_3_2'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 3, 2))
df_all['w2v_search_desc_3_3'] = df_all['product_info'].map(lambda x:get_gram_ratio(w2v, x.split('\t')[0],x.split('\t')[2], 3, 3))
print("--- W2V: %s minutes ---" % round(((time.time() - start_time)/60),2))
start_time = time.time()


df_all.fillna(0, inplace=True)

df_brand = pd.unique(df_all.brand.ravel())
d={}
i = 1
for s in df_brand:
    d[s]=i
    i+=1
df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])

df_material = pd.unique(df_all.material.ravel())
d={}
i = 1
for s in df_material:
    d[s]=i
    i+=1
df_all['material_feature'] = df_all['material'].map(lambda x:d[x])

df_color = pd.unique(df_all.color.ravel())
d={}
i = 1
for s in df_color:
    d[s]=i
    i+=1
df_all['color_feature'] = df_all['color'].map(lambda x:d[x])


df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))
#df_all.to_csv('df_all.csv')
#df_all = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)
df_train = df_all.iloc[:num_train]
df_train.to_csv('X_train_test_script_1.csv')
df_test = df_all.iloc[num_train:]

id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train[:]
X_test = df_test[:]
print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))

rfr = RandomForestRegressor(n_estimators = 100, random_state = 2016, verbose = 1)
xgbr = XGBRegressor(nthread=1)

tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
tsvd = TruncatedSVD(n_components=50, random_state = 2016)

rbm = BernoulliRBM(verbose=True)
norm = Normalizer()
min_max = MinMaxScaler()

pca = PCA(n_components=0.99)

clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                        ('txt5', pipeline.Pipeline([('s5', cust_txt_col(key='material')), ('tfidf5', tfidf), ('tsvd5', tsvd)])),
                        ('txt6', pipeline.Pipeline([('s6', cust_txt_col(key='color')), ('tfidf6', tfidf), ('tsvd6', tsvd)])),
                        ],
                        n_jobs = 1
                    #transformer_weights = {
                    #    'cst':  1.0,
                    #    'txt1': 1.0,
                    #    'txt2': 1.0,
                    #    'txt3': 1.0,
                    #    'txt4': 0.5,
                    #    'txt5': 0.5,
                    #    'txt6': 0.5,
                    #    },

                )),
        #('pca', pca),
        #('rfr', rfr)])
        #('keras', KerasRegressor(get_keras, batch_size=16, nb_epoch=100, validation_split=0.1, shuffle=True))])
        ('reg', xgbr)])


# [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]
# {'reg__learning_rate': 0.01, 'reg__max_depth': 3, 'reg__n_estimators': 1750}

param_grid = {
               #'union__transformer_weights': [list(x) for x in iter_product([0.5, 1.0], repeat=7)]
               #'union__transformer_weights': [list(x) for x in iter_product([0.0, 0.5, 1.0], repeat=7)],
               'reg__n_estimators': np.linspace(500, 3000, 3, dtype=np.int),
               'reg__max_depth': np.linspace(3, 15, 3, dtype=np.int),
               'reg__learning_rate': np.linspace(0.001, 0.3, 3),
               'reg__subsample': np.linspace(0.3, 1, 3),
               'reg__colsample_bytree': np.linspace(0.3, 1, 3),
               #'pca__whiten' : [True, False]
              }

fit_params = {'reg__eval_metric':'rmse'}

model = grid_search.GridSearchCV(estimator = clf,
                                 param_grid = param_grid,
                                 n_jobs = 20,
                                 cv = 5,
                                 verbose = 5,
                                 scoring=RMSE,
                                 fit_params = fit_params)

#base_cross_val(model, X_train, y_train, fit_params = {'reg__eval_metric':'rmse'})
model.fit(X_train, y_train)
print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
'''
y_pred = model.predict(X_test)

y_pred[y_pred > 3] = 3
y_pred[y_pred < 1] = 1

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission_test_script_20160330.csv',index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))
'''

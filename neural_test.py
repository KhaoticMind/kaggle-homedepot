# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:45:41 2016

@author: ur57
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.layers.normalization import BatchNormalization
from keras.constraints import nonneg

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams

from gensim.models import Word2Vec

import sys

import theano
theano.config.openmp = True

stop = stopwords.words('english')
stem = SnowballStemmer('english').stem

def rmse(y_true, y_pred):
    from keras import backend as k
    from keras.objectives import mean_squared_error

    return k.sqrt(mean_squared_error(y_true, y_pred))


def stemmer(sent):
    return ' '.join([stem(word)
                    for word in word_tokenize(sent.lower())
                    if word not in stop])


def get_gram_ratio(text1, text2, w2v, n_grams=1, w=100, h=5700):
    arr = np.ndarray((w, h), np.float)
    arr.fill(-1)
    t1 = list(ngrams(text1.split(), n_grams))
    t2 = list(ngrams(text2.split(), n_grams))
    for i in range(len(t1)):
        for j in range(len(t2)):
            try:
                arr[i, j] = w2v.n_similarity(t1[i], t2[j])
            except:
                pass
    return arr


df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")[:100]
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1")[:10]
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

df_all.fillna('', inplace=True)

df_all['search_term'] = df_all['search_term'].map(lambda x: stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: stemmer(x))
df_all['brand'] = df_all['brand'].map(lambda x: stemmer(x))
df_all['material'] = df_all['material'].map(lambda x: stemmer(x))
df_all['color'] = df_all['color'].map(lambda x: stemmer(x))

id_test = df_test['id']
y_train = df_train['relevance'].values

del df_color, df_pro_desc, df_attr, df_brand, df_material


sent = df_all['product_description'].str.split().tolist()
w2v = Word2Vec(sent, workers=5)
words_importance = []
for row in df_all[['search_term', 'product_description']].itertuples():
    arr1 = get_gram_ratio(row.search_term, row.product_description, w2v)
    arr2 = get_gram_ratio(row.search_term, row.product_description, w2v, 2)
    arr3 = get_gram_ratio(row.search_term, row.product_description, w2v, 3)
    arr4 = get_gram_ratio(row.search_term, row.product_description, w2v, 4)

    words_importance.append(np.stack([arr1, arr2, arr3, arr4]))

words_importance = np.asarray(words_importance)

letter_relation = []
for row in df_all.itertuples():
    arr = np.ndarray((4, 100, 5700), dtype=np.byte)
    arr.fill(0)
    for i in range(len(row.search_term)):
        val_i = ord(row.search_term[i])

        for j in range(len(row.product_description)):
            val_j = ord(row.product_description[j])
            arr[0, i, j] = abs(255 - val_i - val_j)

        for j in range(len(row.brand)):
            val_j = ord(row.brand[j])
            arr[1, i, j] = abs(255 - val_i - val_j)

        for j in range(len(row.material)):
            val_j = ord(row.material[j])
            arr[2, i, j] = abs(255 - val_i - val_j)

        for j in range(len(row.color)):
            val_j = ord(row.color[j])
            arr[3, i, j] = abs(255 - val_i - val_j)

    letter_relation.append(arr)

letter_relation = np.asarray(letter_relation)

X = np.concatenate((words_importance, letter_relation), axis=1)
X_train = X[:num_train]
X_test = X[num_train:]

del df_all, X

model = Sequential()
model.add(BatchNormalization(input_shape=(8, 100, 5700), mode=0, axis=1))
model.add(Convolution2D(16, 8, 8, border_mode='valid'))
model.add(Activation('sigmoid'))
model.add(Convolution2D(16, 8, 8))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model = Sequential()
#model.add(Merge([tri, quad], mode='concat'))

model.add(Convolution2D(32, 4, 4, border_mode='valid'))
model.add(Activation('sigmoid'))
model.add(Convolution2D(32, 4, 4))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1, W_constraint=nonneg()))
model.add(Activation('linear'))

model.compile(loss=rmse, optimizer='sgd')

model.fit(X_train, y_train, batch_size=16, nb_epoch=10, validation_split=0.1)

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:45:41 2016

@author: ur57
"""

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.constraints import nonneg
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import LearningRateScheduler, EarlyStopping

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import ngrams

from gensim.models import Word2Vec

from itertools import product as iter_product

from sklearn.cross_validation import train_test_split

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


def rebin(ndarray, new_shape, operation='avg'):
    """
    https://gist.github.com/derricw/95eab740e1b08b78c03f
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray


def get_gram_ratio(text1, text2, w2v, n_grams_1=1, n_grams_2=1, w=30, h=2000):
    arr = np.ndarray((w, h), np.float32)
    arr.fill(0)
    t1 = list(ngrams(text1.split(), n_grams_1))
    t2 = list(ngrams(text2.split(), n_grams_2))
    for i in range(len(t1)):
        for j in range(len(t2)):
            try:
                arr[i, j] = w2v.n_similarity(t1[i], t2[j])
            except:
                pass
    return arr


def data_gen(df_all, y_train=None, n_batch=5, loop_forever=True):
    res = []
    y_res = []
    while True:
        pos = 0
        for row in df_all.itertuples():
            search_term = str(row.search_term)
            bag = str(row.bag)
            arr = np.ndarray((1, 64, 4800), np.float32)
            arr.fill(128)
            for i in range(len(search_term )):
                val_i = ord(search_term[i])

                for j in range(len(bag)):
                    val_j = ord(bag[j])
                    arr[0, i, j] = abs(val_i - val_j)

            arr = arr.reshape((1, 640, 480))
            res.append(arr)
            if y_train is not None:
                y_res.append(y_train[pos])
                pos += 1

            if len(res) == n_batch:
                res = np.asarray(res)
                if y_train is not None:
                    y_res = np.asarray(y_res)
                    yield (res, y_res)
                    y_res = []
                else:
                    yield (res)

                res = []

        if not loop_forever:
            if len(res) > 0:
                res = np.asarray(res)
                if y_train is not None:
                    y_res = np.asarray(y_res)
                    yield (res, y_res)
                    y_res = []
                else:
                    yield (res)

            break


def batch_test(model, data_x, data_y, n_batch=5):
    foo = data_gen(data_x, data_y, n_batch=n_batch, loop_forever=False)
    res = []
    for x, y in foo:
        ans = model.test_on_batch(x, y)
        res.extend(ans)
        print('{:5.4f} - {:5.4f}'.format(np.mean(res), np.std(res)))
    return res


def batch_predict(model, data, n_batch=5):
    foo = data_gen(data, n_batch=n_batch, loop_forever=False)
    total = data.shape[0]
    res = []
    for x in foo:
        ans = model.predict(x).ravel().tolist()
        res.extend(ans)
        print('{:d} de {:d}'.format(len(res), total))
    return res


'''
df_train = pd.read_csv('train.csv', encoding="ISO-8859-1", nrows=None)
df_test = pd.read_csv('test.csv', encoding="ISO-8859-1", nrows=None)
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

id_test = df_test['id']
y_train = df_train['relevance'].values

del df_color, df_attr, df_brand, df_material, df_train, df_test

df_all.fillna('', inplace=True)

df_all['search_term'] = df_all['search_term'].map(lambda x: stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x: stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: stemmer(x))
df_all['brand'] = df_all['brand'].map(lambda x: stemmer(x))
df_all['material'] = df_all['material'].map(lambda x: stemmer(x))
df_all['color'] = df_all['color'].map(lambda x: stemmer(x))

df_all['bag'] = df_all['product_title'].str.pad(df_all['product_title'].str.len().max(), side='right') + ' ' + \
                df_all['product_description'].str.pad(df_all['product_description'].str.len().max(), side='right') + ' ' + \
                df_all['brand'].str.pad(df_all['brand'].str.len().max(), side='right') + ' ' + \
                df_all['material'].str.pad(df_all['material'].str.len().max(), side='right') + ' ' + \
                df_all['color'].str.pad(df_all['color'].str.len().max(), side='right')
'''


'''
sent = df_pro_desc['product_description'].str.split().tolist()
del df_pro_desc

w2v = Word2Vec(sent, workers=5)
words_importance = []
arrs = []
for row in df_all[['search_term', 'product_description']].itertuples():
    arrs = []
    for (n1, n2) in list(iter_product([1, 2, 3], repeat=2)):
        arr = get_gram_ratio(row.search_term, row.product_description, w2v, n1, n2)
        arrs.append(arr)

    words_importance.append(np.stack(arrs))

words_importance = np.asarray(words_importance)

'''

#X = np.concatenate((words_importance, letter_relation), axis=1)

#df_train = df_all[:num_train]
#df_test = df_all[num_train:]
#df_train.to_csv('processed_train.csv')
#df_test.to_csv('processed_test.csv')

df_train = pd.read_csv('processed_train.csv', index_col=0, nrows=500)
df_test = pd.read_csv('processed_test.csv', index_col=0, nrows=100)
y_train = pd.read_csv('y_train.csv', index_col=0, header=None, nrows=500).values.ravel()


model = Sequential()
model.add(BatchNormalization(input_shape=(1, 640, 480)))
model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(16, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(24, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(24, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(32, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(40, 3, 3, border_mode='valid'))
model.add(LeakyReLU())
model.add(Convolution2D(40, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU())
model.add(Dense(256))
model.add(LeakyReLU())
model.add(Dropout(0.50))

model.add(Dense(1, W_constraint=nonneg()))
model.add(Activation('linear'))

model.compile(loss=rmse, optimizer='rmsprop')

#model.load_weights('my_model_4cnn_2hl_weigths.h5')

def learn_reducer(epoch):
    return 0.01 / (3 *(epoch + 1))

X_train = df_train
#X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.1)

model.fit_generator(data_gen(X_train, y_train, n_batch=10),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=10,
                    callbacks=[
                               #LearningRateScheduler(learn_reducer),
                               EarlyStopping(patience=3, mode='min', monitor='loss')
                               ],
                    nb_worker=4,
                    #validation_data=data_gen(X_test, y_test),
                    #nb_val_samples=X_test.shape[0]
                    )

id_test = df_test['id']
print(batch_predict(model, df_test))

#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('neural_submission.csv',index=False)

#model.save_weights('my_model_3cnn_2hl_weigths.h5')

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
import theano
theano.config.openmp = True



def rmse(y_true, y_pred):
    from keras import backend as k
    from keras.objectives import mean_squared_error

    return k.sqrt(mean_squared_error(y_true, y_pred))


df_train = pd.read_csv('train.csv', encoding="ISO-8859-1")[:10000]
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

df_all['search_term'] = df_all['search_term'].str.lower()
df_all['product_description'] = df_all['product_description'].str.lower()
df_all['brand'] = df_all['brand'].str.lower()
df_all['material'] = df_all['material'].str.lower()
df_all['color'] = df_all['color'].str.lower()

df_all.fillna('', inplace=True)

df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]

id_test = df_test['id']
y_train = df_train['relevance'].values
X_train = []
X_test = []

for row in df_train.itertuples():
    arr = np.ndarray((4, 65, 5500))
    arr.fill(255 * 255)
    for i in range(len(row.search_term)):
        val_i = ord(row.search_term[i])

        for j in range(len(row.product_description)):
            val_j = ord(row.product_description[j])
            arr[0, i, j] = abs(val_i - val_j) * 255

        for j in range(len(row.brand)):
            val_j = ord(row.brand[j])
            arr[1, i, j] = abs(val_i - val_j) * 255

        for j in range(len(row.material)):
            val_j = ord(row.material[j])
            arr[2, i, j] = abs(val_i - val_j) * 255

        for j in range(len(row.color)):
            val_j = ord(row.color[j])
            arr[3, i, j] = abs(val_i - val_j) * 255

    X_train.append(arr)

X_train = np.asarray(X_train)

model = Sequential()
model.add(BatchNormalization(input_shape=(4, 55, 5500), mode=0, axis=1))
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

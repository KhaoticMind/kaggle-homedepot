# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from sklearn.feature_selection import VarianceThreshold

import joblib

from gensim.models import Word2Vec

import re
from itertools import product as iter_product
import time

import helper_processing

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import Parallel, delayed

def load_data():
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

    df_all.fillna(' ', inplace=True)
    return df_all, num_train

def _tokenizer(s):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) ##'desgruda' palavras que estão juntas

        s = s.lower()

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
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = s.replace("  +"," ")

        s = " ".join([word for word in word_tokenize(s)])
        return s
    else:
        return str(s)


def _stemmer(s, stemmer):

    return (" ").join([stemmer.stem(word) for word in s.split()
                        if word not in stop])

def stemmer(df_all, cols):
    df = pd.DataFrame()
    stemmer = SnowballStemmer('english')

    for col in cols:
        df[col + '_stemed'] = df_all[col].map(lambda x:_stemmer(x, stemmer))

    return df

def tokenizer(df_all, cols):
    df = pd.DataFrame()

    for col in cols:
        df[col + '_tokenized'] = df_all[col].map(lambda x:_tokenizer(x))

    return df


def correct_phrase(phrase):
    return ' '.join([correct(x.lower()) for x in phrase.split()])


def spell_correct(df_all, cols):
    df = pd.DataFrame()

    for col in cols:
        df[col + '_corrected'] = df_all[col].map(lambda x: correct_phrase(x))

    return df

def _get_search_col(cols):
    for col in cols:
        if 'search_term' in col:
            return col

def char_len_and_ratio(df_all, cols):
    df = pd.DataFrame()

    search_col = _get_search_col(cols)
    cols = list(set(cols) - set([search_col]))

    df['char_len_of_' + search_col] = df_all[search_col].map(lambda x:len(list(x)) + 1).astype(np.int64)

    for col in cols:
        df['char_len_of_' + col] = df_all[col].map(lambda x:len(list(x))).astype(np.int64)
        df['char_ratio_of_' + col] = df['char_len_of_' + col] / df['char_len_of_' + search_col]

    return df


def word_len_and_ratio(df_all, cols):
    df = pd.DataFrame()
    search_col = _get_search_col(cols)
    cols = list(set(cols) - set([search_col]))

    df['word_len_of_' + search_col] = df_all[search_col].map(lambda x:len(x.split()) + 1).astype(np.int64)

    for col in cols:
        df['word_len_of_' + col] = df_all[col].map(lambda x:len(x.split())).astype(np.int64)
        df['word_ratio_of_' + col] = df['word_len_of_' + col] / df['word_len_of_' + search_col]

    return df


def gram_len_and_ratio(df_all, cols):

    df = pd.DataFrame()
    search_col = _get_search_col(cols)
    cols = list(set(cols) - set([search_col]))

    for i in range(6):
        gram_size = i + 1
        df[str(gram_size) + 'grams_in_' + search_col] = df_all[search_col].map(lambda x: len(list(ngrams(x, gram_size))) + 1)
        for col in cols:
            df[str(gram_size) + 'grams_in_' + col] = df_all[col].map(lambda x: len(list(ngrams(x, gram_size))))
            df[str(gram_size) + 'grams_ratio_' + col] = df[str(gram_size) + 'grams_in_' + col] /  df[str(gram_size) + 'grams_in_' + search_col]

    return df


def average_edit_distance(str1, str2, metric):
    acc = 0
    for a in str1.split():
        for b in str2.split():
            if metric == nltk.jaccard_distance or metric == nltk.masi_distance:
                acc += metric(set(list(a)), set(list(b)))
            else:
                acc += metric(a, b)


    return len(str1) * (acc / (1 + len(str2)))


def edit_distance(df_all, cols):
    df = pd.DataFrame()
    search_col = _get_search_col(cols)
    cols = list(set(cols) - set([search_col]))

    for col in cols:
        df['levenshtein_distance_' + col] = df_all.apply(axis=1, func=lambda x: average_edit_distance(x[search_col], x[col], nltk.edit_distance))
        df['binary_distance_' + col] = df_all.apply(axis=1, func=lambda x: average_edit_distance(x[search_col], x[col], nltk.binary_distance))
        df['jaccard_distance_' + col] = df_all.apply(axis=1, func=lambda x: average_edit_distance(x[search_col], x[col], nltk.jaccard_distance))
        df['masi_distance_' + col] = df_all.apply(axis=1, func=lambda x: average_edit_distance(x[search_col], x[col], nltk.masi_distance))

    return df


def similarity(w2v, grams):
    try:
        res = w2v.n_similarity(grams[0], grams[1])
        return res
    except:
        return 0


def get_w2v_ratio(w2v, text1, text2, n_grams_1=1, n_grams_2=1):
    t1 = list(ngrams(text1.split(), n_grams_1))
    t2 = list(ngrams(text2.split(), n_grams_2))
    pairs = iter_product(t1, t2, repeat=1)
    pairs = list(pairs)
    res = list(map(lambda x: similarity(w2v, x), pairs))
    if len(res) == 0:
        return 0
    else:
        return np.mean(res)


def w2v(df_all, cols):
    w2v = Word2Vec.load_word2vec_format('word2vec-GoogleNews-vectors-negative300.bin.gz', binary=True)
    print("Loaded w2v file")
    df = pd.DataFrame()
    search_col = _get_search_col(cols)
    cols = list(set(cols) - set([search_col]))

    for col in cols:
        for i in range(3):
            for j in range(3):
                df['w2v_' + col + str(i) + '_' + str(j)] = df_all.apply(axis=1, func=lambda x: get_w2v_ratio(w2v, x[search_col], x[col], i, j))


def tfidf(df_all, cols):
    df = pd.DataFrame()

    N_FEATURES = 50

    tfidf = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
    tsvd = TruncatedSVD(n_components=N_FEATURES, random_state=2016)
    for col in cols:
        colnames = ['tfidf_' + col + '_' + str(i) for i in range(N_FEATURES)]
        res = tsvd.fit_transform(tfidf.fit_transform(df_all[col]))
        for i in range(N_FEATURES):
            df[colnames[i]] = res[:, i]

    return df


class Memoize:
    """Memoize(fn) - an instance which acts like fn but memoizes its arguments
       Will only work on functions with non-mutable arguments
       https://code.activestate.com/recipes/52201/
    """
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.fn(*args)
        return self.memo[args]


if __name__ == '__main__':

    correct = Memoize(helper_processing.correct)
    stop = stopwords.words('english')

    cols = ['search_term', 'product_title', 'product_description', 'brand', 'material', 'color']
    start_time = time.time()

    #df_all, num_train = load_data()
    #joblib.dump(df_all, 'df_all.pkl', compress=9)
    #joblib.dump(num_train, 'num_train.pkl', compress=9)
    df_all = joblib.load('df_all.pkl')
    num_train = joblib.load('num_train.pkl')
    print("--- Load Data: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_tokenized = tokenizer(df_all, cols)
    #joblib.dump(df_tokenized, 'df_tokenized.pkl', compress=9)
    df_tokenized = joblib.load('df_tokenized.pkl')
    print("--- Tokenization: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_corrected = spell_correct(df_tokenized, df_tokenized.columns)
    #joblib.dump(df_corrected, 'df_corrected.pkl', compress=9)
    df_corrected = joblib.load('df_corrected.pkl')
    print("--- Correction: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_stemmed = stemmer(df_corrected, df_corrected.columns)
    #joblib.dump(df_stemmed, 'df_stemmed.pkl', compress=9)
    df_stemmed = joblib.load('df_stemmed.pkl')
    print("--- Stemmer: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()


    ############
    # Metricas com valores SEM stem
    ############

    #df_char_nstem = char_len_and_ratio(df_corrected, df_corrected.columns)
    #joblib.dump(df_char_nstem, 'df_char_nstem.pkl', compress=9)
    df_char_nstem = joblib.load('df_char_nstem.pkl')
    print("--- Char len s/ stem: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_word_nstem = word_len_and_ratio(df_corrected, df_corrected.columns)
    #joblib.dump(df_word_nstem, 'df_word_nstem.pkl', compress=9)
    df_word_nstem = joblib.load('df_word_nstem.pkl')
    print("--- Word len s/ stem: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_gram_nstem = gram_len_and_ratio(df_corrected, df_corrected.columns)
    #joblib.dump(df_gram_nstem, 'df_gram_nstem.pkl', compress=9)
    df_gram_nstem = joblib.load('df_gram_nstem.pkl')
    print("--- Gram s/ stem: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_edit_nstem = edit_distance(df_corrected, df_corrected.columns)
    #joblib.dump(df_edit_nstem, 'df_edit_nstem.pkl', compress=9)
    df_edit_nstem = joblib.load('df_edit_nstem.pkl')
    print("--- Edit s/ stem: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_tdif_nstem = tfidf(df_corrected, df_corrected.columns)
    #joblib.dump(df_tdif_nstem, 'df_tdif_nstem.pkl', compress=9)
    df_tdif_nstem = joblib.load('df_tdif_nstem.pkl')
    print("--- TDIDF s/ stem: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_w2v_nstem = w2v(df_corrected, df_corrected.columns)
    #joblib.dump(df_w2v_nstem, 'df_w2v_nstem.pkl', compress=9)
    df_w2v_nstem = joblib.load('df_w2v_nstem.pkl')
    print("--- W2V s/ stem: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()


    ############
    # Metricas com valores COM stem
    ############
    #df_char = char_len_and_ratio(df_stemmed, df_stemmed.columns)
    #joblib.dump(df_char, 'df_char.pkl', compress=9)
    df_char = joblib.load('df_char.pkl')
    print("--- Char len: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_word = word_len_and_ratio(df_stemmed, df_stemmed.columns)
    #joblib.dump(df_word, 'df_word.pkl', compress=9)
    df_word = joblib.load('df_word.pkl')
    print("--- Word len: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_gram = gram_len_and_ratio(df_stemmed, df_stemmed.columns)
    #joblib.dump(df_gram, 'df_gram.pkl', compress=9)
    df_gram = joblib.load('df_gram.pkl')
    print("--- Gram: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_edit = edit_distance(df_stemmed, df_stemmed.columns)
    #joblib.dump(df_edit, 'df_edit.pkl', compress=9)
    df_edit = joblib.load('df_edit.pkl')
    print("--- Edit: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    #df_tdif = tfidf(df_stemmed, df_stemmed.columns)
    #joblib.dump(df_tdif, 'df_tdif.pkl', compress=9)
    df_tdif = joblib.load('df_tdif.pkl')
    print("--- TDIDF : %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    df_w2v = w2v(df_stemmed, df_stemmed.columns)
    joblib.dump(df_w2v, 'df_w2v.pkl', compress=9)
    #df_w2v = joblib.load('df_w2v.pkl')
    print("--- W2V: %s minutes ---" % round(((time.time() - start_time)/60),2))
    start_time = time.time()

    df_tudo = pd.concat((df_all,
                         df_corrected,
                         df_stemmed,
                         df_char,
                         df_word,
                         df_gram,
                         df_edit,
                         df_w2v,
                         df_tdif,
                         df_char_nstem,
                         df_word_nstem,
                         df_gram_nstem,
                         df_edit_nstem,
                         df_w2v_nstem,
                         df_tdif_nstem,
                        ), axis=1)

    del df_all, df_corrected, df_stemmed
    joblib.dump(df_tudo, 'df_tudo.pkl', compress=9)

    df_metrics = pd.concat((df_char,
                            df_word,
                            df_gram,
                            df_edit,
                            df_w2v,
                            df_tdif,
                            df_char_nstem,
                            df_word_nstem,
                            df_gram_nstem,
                            df_edit_nstem,
                            df_w2v_nstem,
                            df_tdif_nstem,
                          ), axis=1).replace(np.inf, 1e20)  # Some infinites appear on the middle of the data...

    del df_w2v, df_char_nstem, df_word_nstem, df_gram_nstem, df_w2v_nstem
    del df_gram, df_edit, df_tdif, df_tdif_nstem, df_char, df_word,
    joblib.dump(df_metrics, 'df_metrics.pkl', compress=9)

    #id_test = df_tudo.iloc[num_train:]['id']
    #y_train = df_tudo.iloc[:num_train].relevance.values
    #joblib.dump(y_train, 'y_train.pkl')
    #joblib.dump(id_test, 'id_test.pkl')
    id_test = joblib.load('id_test.pkl')
    y_train = joblib.load('y_train.pkl')

    var = VarianceThreshold()
    var.fit_transform(df_metrics)
    df_val_metrics = df_metrics[var.get_support(indices=True)]
    joblib.dump(df_val_metrics, 'df_val_metrics.pkl', compress=9)

    df_train = df_val_metrics.iloc[:num_train]
    df_test = df_val_metrics.iloc[num_train:]

# -*- coding: utf-8 -*-


#  -----------------------------------------
# http://norvig.com/spell-correct.html
#  -----------------------------------------
import re, collections
from gensim.models import Word2Vec
from nltk import ngrams
import numpy as np

from nltk.corpus import inaugural, reuters, brown, gutenberg

from itertools import product as iter_product

def words(text):
    return re.findall('[a-z]+', text.lower())


def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(inaugural.raw() + reuters.raw() + brown.raw() + gutenberg.raw()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts    = [a + c + b     for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


def known(words):
    return set(w for w in words if w in NWORDS)


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

#  -----------------------------------------
# By myself
#  -----------------------------------------

def similarity(w2v, grams):
    try:
        res = w2v.n_similarity(grams[0], grams[1])
        return res
    except:
        return 0

def get_gram_ratio(w2v, text1, text2, n_grams_1=1, n_grams_2=1, n_jobs=1):
    t1 = list(ngrams(text1.split(), n_grams_1))
    t2 = list(ngrams(text2.split(), n_grams_2))
    pairs = list(iter_product(t1, t2, repeat=1))
    res = list(map(lambda x: similarity(w2v, x), pairs))
    if len(res) == 0:
        return 0
    else:
        return np.mean(res)

'''
sent = df_pro_desc['product_description'].str.split().tolist()
del df_pro_desc


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
# --------------------------------------
# https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos
# --------------------------------------
import requests
import re
import time
from random import randint

START_SPELL_CHECK="<span class=\"spell\">Showing results for</span>"
END_SPELL_CHECK="<br><span class=\"spell_orig\">Search instead for"

HTML_Codes = (
        ("'", '&#39;'),
        ('"', '&quot;'),
        ('>', '&gt;'),
        ('<', '&lt;'),
        ('&', '&amp;'),
)


def spell_check(s, proxies=None):
    q = '+'.join(s.split())
    time.sleep(randint(0, 2))  # relax and don't let google be angry
    r = requests.get("https://www.google.com.br/search?q="+q, proxies=proxies)
    content = r.text
    start = content.find(START_SPELL_CHECK)
    if (start > -1):
        start = start + len(START_SPELL_CHECK)
        end = content.find(END_SPELL_CHECK)
        search = content[start:end]
        search = re.sub(r'<[^>]+>', '', search)
        for code in HTML_Codes:
            search = search.replace(code[1], code[0])
        search = search[1:]
    else:
        search = s
    return search

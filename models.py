import shutil
import tarfile
import zipfile
from os.path import join, exists
from os import mkdir
import sys

import gensim
import wget
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from pytorch_pretrained_bert import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel

PATH = 'models'

MODELS = ['word2vec', 'glove', 'dict2vec', 'conceptnet', 'bert', 'gpt2']
MODEL_PATH_DICT = {'word2vec': 'GoogleNews-vectors-negative300.bin.gz',
                   'glove': 'glove_gensim.txt',
                   'dict2vec': 'dict2vec-vectors-dim300.vec',
                   'conceptnet': 'numberbatch-en-19.08.txt.gz',
                   'bert': 'NOPATH',
                   'gpt2': 'NOPATH'}


def download_model(name):
    if not name in MODELS:
        raise Exception(str(name) + ' not a model in the list')
    if not exists(PATH):
        print("# ", str(PATH), "not found, creating dir.")
        mkdir(PATH)
    print('# Downloading model: ' + str(name))
    name_path = MODEL_PATH_DICT[name]
    if name == 'word2vec':
        if not exists(join(PATH, name_path)):
            wget.download('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')
            shutil.move(name_path, join(PATH, name_path))
            print('# Downloaded word2vec')
        else:
            print('# Already downloaded')
    if name == 'glove':
        if not exists(join(PATH, name_path)):
            wget.download('http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip')
            zip = zipfile.ZipFile('./glove.840B.300d.zip')
            zip.extractall()
            _ = glove2word2vec('./glove.840B.300d.txt', join(PATH, name_path))
            print('# Downloaded glove')
        else:
            print('# Already downloaded')
    if name == 'dict2vec':
        if not exists(join(PATH, name_path)):
            wget.download('https://dict2vec.s3.amazonaws.com/dict2vec300.tar.bz2')
            tar = tarfile.open("dict2vec300.tar.bz2")
            tar.extractall()
            tar.close()
            shutil.move(name_path, join(PATH, name_path))
            print('# Downloaded dict2vec')
        else:
            print('# Already downloaded')

    if name == 'conceptnet':
        if not exists(join(PATH, name_path)):
            wget.download('https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz')
            shutil.move(name_path, join(PATH, name_path))
            print('# Downloaded Conceptnet Numberbatch')
        else:
            print('# Already downloaded')
    if name == 'bert':
        _ = BertTokenizer.from_pretrained('bert-large-uncased')
        _ = BertModel.from_pretrained('bert-large-uncased').embeddings.word_embeddings.weight.data.numpy()
        print('# Downloaded bert')
    if name == 'gpt2':
        _ = GPT2Tokenizer.from_pretrained('gpt2')
        _ = GPT2LMHeadModel.from_pretrained('gpt2').transformer.wte.weight.data.numpy()
        print('# Downloaded gpt-2')


def download_all():
    for name in MODELS:
        download_model(name)
    print('# Successfully downloaded all models')


def load_model_fromlist(name):
    if not name in MODELS:
        raise Exception(str(name) + ' not a model in the list')
    print('# Loading model: ' + str(name))
    name_path = MODEL_PATH_DICT[name]
    if name == 'word2vec':
        if not exists(join(PATH, name_path)): download_model(name)
        return (gensim.models.KeyedVectors.load_word2vec_format(join(PATH, name_path), binary=True))
    if name == 'glove':
        if not exists(join(PATH, name_path)): download_model(name)
        return (gensim.models.KeyedVectors.load_word2vec_format(join(PATH, name_path)))
    if name == 'dict2vec':
        if not exists(join(PATH, name_path)): download_model(name)
        return (gensim.models.KeyedVectors.load_word2vec_format(join(PATH, name_path), binary=False,
                                                                unicode_errors="ignore"))
    if name == 'conceptnet':
        if not exists(join(PATH, name_path)): download_model(name)
        return (gensim.models.KeyedVectors.load_word2vec_format(join(PATH, name_path)))
    if name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased').embeddings.word_embeddings.weight.data.numpy()
        return ([model, tokenizer])
    if name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2').transformer.wte.weight.data.numpy()
        return ([model, tokenizer])


def load_model_custom(model_path, binary=False):
    if not exists(join(PATH, model_path)):
        raise Exception('# Model not found in /models: ', model_path)
    return (gensim.models.KeyedVectors.load_word2vec_format(join(PATH, model_path), binary=binary))

def load_model(name, binary=False):
    if name in MODELS:
        return(load_model_fromlist(name))
    else:
        return(load_model_custom(name, binary))

def vocabulary_model(model):
    vocabulary_keys = model.wv.vocab.keys()
    vocabulary_set = set(vocabulary_keys)  # , np.array(list(vocabulary_keys))
    return (vocabulary_set)


def clean_pairs_fromvocab(vocabulary_set, pairs_sets):
    pairs_sets_clean = [[d for d in list(pairs_sets[i]) if d[0] in vocabulary_set and d[1] in vocabulary_set] for i
                        in range(len(pairs_sets))]
    return (pairs_sets_clean)


def clean_pairs(model, pairs_sets):
    if type(model) == list:
        return (pairs_sets)

    vocabulary_set = vocabulary_model(model)
    return (clean_pairs_fromvocab(vocabulary_set, pairs_sets))


if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) == 1:
        download_all()
    else:
        name = sys.argv[1]
        download_model(name)

# word2vec
# wget.download('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')

# Glove
# !wget http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip
# !unzip glove.840B.300d.zip

# dict2vec
# !wget https://dict2vec.s3.amazonaws.com/dict2vec300.tar.bz2
# !tar -x dict2vec300.tar.bz2

# ConceptNet Numberbatch
# !wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz

# BERT and GPT-2 ==> Loading will download them
# bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# bert_model = BertModel.from_pretrained('bert-large-uncased').embeddings.word_embeddings.weight.data.numpy()

# gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').transformer.wte.weight.data.numpy()


# word2vec
# pretrained_embeddings_path = "word-embeddings-geometry/GoogleNews-vectors-negative300.bin.gz"
# "./GoogleNews-vectors-negative300.bin.gz"
# model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path,
# binary=True)

# Glove
# tmp_file = get_tmpfile("./glove_gensim.txt")
# _ = glove2word2vec('./glove.840B.300d.txt', tmp_file)

# model_glove = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)

# dict2vec
# model_dict2vec = gensim.models.KeyedVectors.load_word2vec_format("dict2vec-vectors-dim300.vec", binary=False, unicode_errors="ignore")

# ConceptNet Numberbatch
# pretrained_embeddings_path = "numberbatch-en-19.08.txt.gz"
# model_conceptnet = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path)

# BERT and GPT-2
# bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# bert_model = BertModel.from_pretrained('bert-large-uncased').embeddings.word_embeddings.weight.data.numpy()

# gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').transformer.wte.weight.data.numpy()
'''
import gensim
from gensim import utils, matutils
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec, Word2VecKeyedVectors

import logging
import wget
from itertools import chain
import logging
from six import string_types
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import os
import sklearn

import scipy
from scipy import sparse
from scipy.stats import ttest_ind
from scipy.sparse.linalg import norm
from scipy.stats import iqr

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering

from svd2vec import svd2vec

import plotly.graph_objects as go
import plotly

import tensorflow
import transformers
'''

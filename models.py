import shutil
import tarfile
import zipfile
from os.path import join, exists
from os import mkdir
import sys

import gensim
import wget
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.word2vec import Word2Vec
from pytorch_pretrained_bert import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model


PATH = 'models'

MODELS = ['word2vec', 'glove', 'dict2vec', 'conceptnet', 'bert', 'gpt2', 'bert-context', 'gpt2-context']
MODEL_PATH_DICT = {'word2vec': 'GoogleNews-vectors-negative300.bin.gz',
                   'glove': 'glove_gensim.txt',
                   'dict2vec': 'dict2vec-vectors-dim300.vec',
                   'conceptnet': 'numberbatch-en-19.08.txt.gz',
                   'bert': 'NOPATH',
                   'gpt2': 'NOPATH',
                   'bert-context': 'NOPATH',
                   'gpt2-context': 'NOPATH'}


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
    if name == 'bert' or name == 'bert-context':
        _ = BertTokenizer.from_pretrained('bert-large-uncased')
        _ = BertModel.from_pretrained('bert-large-uncased').embeddings.word_embeddings.weight.data.numpy()
        print('# Downloaded bert')
    if name == 'gpt2' or name == 'gpt2-context':
        _ = GPT2Tokenizer.from_pretrained('gpt2')
        _ = GPT2LMHeadModel.from_pretrained('gpt2')
        _ = GPT2Model.from_pretrained('gpt2')
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
    if name == 'bert-context':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states = True)
        return ([model, tokenizer])
    if name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2').transformer.wte.weight.data.numpy()
        return ([model, tokenizer])
    if name == 'gpt2-context':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2', output_hidden_states = True)
        return ([model, tokenizer])

def load_model_custom(model_path, binary=False):
    print("# Loading custom model: ", model_path)
    if not exists(join(PATH, model_path)):
        raise Exception('# Model not found in /models: ', model_path)
    if model_path[-5:] == 'model':
       return (Word2Vec.load(join(PATH, model_path)))
    return (gensim.models.KeyedVectors.load_word2vec_format(join(PATH, model_path), binary=binary))

def load_model(name, binary=False):
    if name in MODELS:
        return(load_model_fromlist(name))
    else:
        return(load_model_custom(name, binary))

def vocabulary_model(model):
    vocabulary_keys = model.vocab.keys()
    vocabulary_set = set(vocabulary_keys)
    return (vocabulary_set)


def clean_pairs_fromvocab(vocabulary_set, pairs_sets):
    pairs_sets_clean = [[d for d in list(pairs_sets[i]) if d[0] in vocabulary_set and d[1] in vocabulary_set] for i
                        in range(len(pairs_sets))]
    return (pairs_sets_clean)


def clean_pairs(model, pairs_sets):
    if type(model) == list:
        return ([list(p) for p in pairs_sets])

    vocabulary_set = vocabulary_model(model)
    return (clean_pairs_fromvocab(vocabulary_set, pairs_sets))


if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) == 1:
        download_all()
    else:
        name = sys.argv[1]
        download_model(name)

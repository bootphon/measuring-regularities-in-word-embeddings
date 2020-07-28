# Using chrisjmccormick's github for the basic word2vec import

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

from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

from pytorch_pretrained_bert import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel


import sys
import pandas as pd
import time

from os.path import exists
from os import mkdir

from read_bats import bats_names_pairs
from models import vocabulary_model, load_model, MODELS

## Version modifiée de gensim, peut être à verifier quels sont les changements exactement (le but est principalement d'autorizer la version vanilla)
def most_similar(model, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None, ignore=True):
    if positive is None:
        positive = []
    if negative is None:
        negative = []

    model.init_sims()

    if isinstance(positive, string_types) and not negative:
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]

    # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
    positive = [
        (word, 1.0) if isinstance(word, string_types + (np.ndarray,)) else word
        for word in positive
    ]
    negative = [
        (word, -1.0) if isinstance(word, string_types + (np.ndarray,)) else word
        for word in negative
    ]

    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, np.ndarray):
            mean.append(weight * word)
        else:
            mean.append(weight * model.word_vec(word, use_norm=True))
            if word in model.vocab:
                all_words.add(model.vocab[word].index)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)

    if indexer is not None:
        return indexer.most_similar(mean, topn)

    limited = model.vectors_norm if restrict_vocab is None else model.vectors_norm[:restrict_vocab]
    dists = np.dot(limited, mean)
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
    # ignore (don't return) words from the input
    if ignore:
        result = [(model.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
    else:
        result = [(model.index2word[sim], float(dists[sim])) for sim in best]
    return result[:topn]


def evaluate_word_analogies_bats(model, directory, restrict_vocab=300000, case_insensitive=True, dummy4unknown=False):
    logger = logging.getLogger(__name__)
    print("# Computing analogy scores for category type: ", str(directory))

    ok_vocab = [(w, model.vocab[w]) for w in model.index2word[:restrict_vocab]]
    ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)
    oov = 0
    # logger.info("Evaluating word analogies for top %i words in the model on %s", restrict_vocab, analogies)
    sections, section = [], None
    quadruplets_no = 0

    directions_names_bats = []
    pairs_sets = []

    scores_bats = [] #dict()
    scores_bats_vanilla = [] #dict()

    for f in os.listdir('BATS_3.0/' + str(directory)): #..
        directions_names_bats.append(str(f)[:-4])
        pairs_sets.append(set())
        with utils.open_file('BATS_3.0/' + str(directory) + '/' + str(f)) as fin:
            for line_no, line in enumerate(fin):
                line = utils.to_unicode(line)
                a, b = [word.lower() for word in line.split()]
                list_b = b.split('/')
                if list_b[0] != a:
                    pairs_sets[-1].add((a.upper(), list_b[0].upper()))

    for i in range(len(directions_names_bats)):
        if section:
            # store the last section, too
            sections.append(section)
            # model._log_evaluate_word_analogies(section)
            correct, incorrect = len(section['correct']), len(section['incorrect'])
            if correct + incorrect > 0:
                score = correct / (correct + incorrect)
                logger.info("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
                scores_bats.append([section['section'], score, correct, correct + incorrect])
            else:
                print('No score for ', section['section'])
            correct, incorrect = len(section['correct_vanilla']), len(section['incorrect_vanilla'])
            if correct + incorrect > 0:
                score = correct / (correct + incorrect)
                logger.info("%s: %.1f%% (%i/%i) VANILLA", section['section'], 100.0 * score, correct,
                            correct + incorrect)
                scores_bats_vanilla.append([section['section'], score, correct, correct + incorrect])
            total_section = len(section['correct_vanilla']) + len(section['incorrect_vanilla'])
            if total_section > 0:
                logger.info('Number of predictions equal to a: %i (%d), a*: %i (%d), b: %i (%d)',
                            section['n_a'], section['n_a'] / total_section,
                            section['n_a*'], section['n_a*'] / total_section,
                            section['n_b'], section['n_b'] / total_section)


        section = {'section': directions_names_bats[i], 'correct': [], 'incorrect': [],
                   'correct_vanilla': [], 'incorrect_vanilla': [], 'n_a': 0, 'n_a*': 0, 'n_b': 0,
                   'cd': [], 'badc': [], 'bac': [], 'n/cba': [], 'n/c': [], 'n/d': []}

        tuples = pairs_sets[i]
        for t1 in tuples:
            for t2 in tuples:
                a, b = t1
                c, expected = t2
                if a != c:
                    quadruplets_no += 1
                    if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                        oov += 1
                        # if dummy4unknown:
                        #    logger.debug('Zero accuracy for line #%d with OOV words: %s', line_no, line.strip())
                        #    section['incorrect'].append((a, b, c, expected))
                        # else:
                        #    logger.debug("Skipping line with OOV words")
                        continue
                    original_vocab = model.vocab
                    model.vocab = ok_vocab

                    predicted = None
                    ignore = {a, b, c}  # input words to be ignored
                    ignore_bool = False
                    positive = [b, c]
                    negative = [a]
                    # find the most likely prediction using 3CosAdd (vector offset) method
                    # TODO: implement 3CosMul and set-based methods for solving analogies
                    sims = most_similar(model, positive=positive, negative=negative, topn=5,
                                        restrict_vocab=restrict_vocab, ignore=ignore_bool)

                    model.vocab = original_vocab

                    # predicted = sims[0][0].upper() if case_insensitive else sims[0][0]
                    for element in sims:
                        predicted = element[0].upper() if case_insensitive else element[0]

                        if predicted in ok_vocab and predicted not in ignore:
                            break
                    for element in sims:
                        predicted_ignore = element[0].upper() if case_insensitive else element[0]

                        if predicted_ignore in ok_vocab:
                            break

                    if predicted == expected:
                        section['correct'].append((a, b, c, expected))
                    else:
                        section['incorrect'].append((a, b, c, expected))

                    if predicted_ignore == expected:
                        section['correct_vanilla'].append((a, b, c, expected))
                    else:
                        section['incorrect_vanilla'].append((a, b, c, expected))
                    if predicted_ignore == a:
                        section['n_a'] += 1
                    if predicted_ignore == b:
                        section['n_a*'] += 1
                    if predicted_ignore == c:
                        section['n_b'] += 1

    if section:
        # store the last section, too
        sections.append(section)
        # model._log_evaluate_word_analogies(section)
        correct, incorrect = len(section['correct']), len(section['incorrect'])
        if correct + incorrect > 0:
            score = correct / (correct + incorrect)
            logger.info("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
            scores_bats.append([section['section'], score, correct, correct + incorrect])
        else:
            print('No score for ', section['section'])
        correct, incorrect = len(section['correct_vanilla']), len(section['incorrect_vanilla'])
        if correct + incorrect > 0:
            score = correct / (correct + incorrect)
            logger.info("%s: %.1f%% (%i/%i) VANILLA", section['section'], 100.0 * score, correct, correct + incorrect)
            scores_bats_vanilla.append([section['section'], score, correct, correct + incorrect])

        total_section = len(section['correct_vanilla']) + len(section['incorrect_vanilla'])
        if total_section > 0:
            logger.info('Number of predictions equal to a: %i (%d), a*: %i (%d), b: %i (%d)',
                        section['n_a'], section['n_a'] / total_section,
                        section['n_a*'], section['n_a*'] / total_section,
                        section['n_b'], section['n_b'] / total_section)

    total = {
        'section': 'Total accuracy',
        'correct': list(chain.from_iterable(s['correct'] for s in sections)),
        'incorrect': list(chain.from_iterable(s['incorrect'] for s in sections)),
        'correct_vanilla': list(chain.from_iterable(s['correct_vanilla'] for s in sections)),
        'incorrect_vanilla': list(chain.from_iterable(s['incorrect_vanilla'] for s in sections)),
    }

    oov_ratio = float(oov) / quadruplets_no * 100
    logger.info('Quadruplets with out-of-vocabulary words: %.1f%%', oov_ratio)
    if not dummy4unknown:
        logger.info(
            'NB: analogies containing OOV words were skipped from evaluation! '
            'To change this behavior, use "dummy4unknown=True"'
        )
    # analogies_score = model._log_evaluate_word_analogies(total)
    correct, incorrect = len(total['correct']), len(total['incorrect'])
    # print(total)
    if correct + incorrect > 0:
        score = correct / (correct + incorrect)
        logger.info("%s: %.1f%% (%i/%i)", total['section'], 100.0 * score, correct, correct + incorrect)
        total_score = ["# Total " + str(directory), score, correct, correct + incorrect]
        analogies_score = score
    correct_vanilla, incorrect_vanilla = len(total['correct_vanilla']), len(total['incorrect_vanilla'])
    # print(total)
    if correct_vanilla + incorrect_vanilla > 0:
        score = correct_vanilla / (correct_vanilla + incorrect_vanilla)
        logger.info("%s: %.1f%% (%i/%i) VANILLA", total['section'], 100.0 * score, correct_vanilla,
                    correct_vanilla + incorrect_vanilla)
        total_score_vanilla = ["# Total " + str(directory), score, correct_vanilla, correct_vanilla + incorrect_vanilla]
        analogies_score = score

    sections.append(total)
    bats_scores = [total_score, total_score_vanilla, scores_bats, scores_bats_vanilla]
    # Return the overall score and the full lists of correct and incorrect analogies
    return bats_scores #[analogies_score, sections, bats_scores]


def bats_test(model):
    results = []
    for d in os.listdir('../BATS_3.0'):
        if d != 'metadata.json':
            results.append(evaluate_word_analogies_bats(model, directory=d))
    return (results)

def save_analogy_test(results):
    total_results = []
    total_results_vanilla = []
    for r in results:
        t, t_v, s, s_v = r[0], r[1], r[2], r[3]
        for si in s:
            total_results.append(si)
        total_results.append(t)
        for si_v in s_v:
            total_results_vanilla.append(si_v)
        total_results_vanilla.append(t_v)

    columns = ['Categories', 'Accuracy', 'Nb correct', 'Nb total']
    df = pd.DataFrame(total_results, columns=columns)
    df_v = pd.DataFrame(total_results_vanilla, columns=columns)

    if not exists('results'):
        print("# ", str('results'), "not found, creating dir.")
        mkdir('results')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    namepath = 'results/' + 'analogy_test' + '-' + str(timestr) + '.csv'
    namepath_v = 'results/' + 'analogy_test_vanilla' + '-' + str(timestr) + '.csv'
    df.to_csv(namepath, index=False)
    df_v.to_csv(namepath_v, index=False)

    print("# Successfully saved the analogy tests to ", str(namepath), "and ", str(namepath_v))

if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) < 2:
        raise("# Please provide a model (all, name, or filename for a custom model)")

    name = sys.argv[1]

    if name == 'all':
        for name in MODELS:
            model = load_model(name)
            print("# Computing the analogy test accuracy from ", str(name))
            results = bats_test(model)
            print("# Sucessfully computed the analogy test accuracy from ", str(name))
            save_analogy_test(results)

    else:
        model = load_model(name)
        print("# Computing the analogy test accuracy from ", str(name))
        results = bats_test(model)
        print("# Sucessfully computed the analogy test accuracy from ", str(name))
        save_analogy_test(results)

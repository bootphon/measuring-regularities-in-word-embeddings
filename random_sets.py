import gensim
from gensim import utils
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

import numpy as np

from metrics import permutation_onecycle, similarite_offsets, normal_and_shuffled_offsets, OCS_PCS
from read_bats import vocab_bats, bats_names_pairs
from models import vocabulary_model, load_model, MODELS


import sys
import pandas as pd
import time

from os.path import exists
from os import mkdir

def offsets_perms_random(model, pairs_sets, vocabulary, nb_random=10, size_random_categ=50, limit_word=10000):
    vocabulary_list = list(vocabulary)
    vocab_used = vocab_bats(pairs_sets)

    print("# Computing random offsets")

    # a* - a, a et a* de la même catégorie mais permuté
    print("# Computing random offsets for permutation within")
    perm_lists_permutation_within = []
    offsets_permutation_within = []
    for k_r in range(nb_random):
        perm_lists_permutation_within.append([])
        offsets_permutation_within.append([])
        for i in range(len(pairs_sets)):
            perm_list = permutation_onecycle(len(pairs_sets[i]))
            offsets_permutation_within[-1].append([])
            ds = list(pairs_sets[i])
            for k in range(len(ds)):
                di = ds[k]
                dj = ds[perm_list[k]]
                if di[0] in vocabulary and dj[1] in vocabulary and dj[1] != di[0]:
                    offsets_permutation_within[-1][-1].append(
                        model.wv.get_vector(dj[1]) - model.wv.get_vector(di[0]))
            perm_lists_permutation_within[-1].append(perm_list)

    print("# Computing random offsets for mismatched within")
    offsets_mismatched_within = []
    perm_lists_mismatched_within = []
    for k_r in range(nb_random):
        perm_list_mismatched_within = np.hstack([permutation_onecycle(10),
                                                 permutation_onecycle((10, 20)),
                                                 permutation_onecycle((20, 30)),
                                                 permutation_onecycle((30, 40)),
                                                 ])
        perm_lists_mismatched_within.append(perm_list_mismatched_within)
        offsets_mismatched_within.append([])

        for i in range(len(pairs_sets)):
            offsets_mismatched_within[-1].append([])
            j = perm_list_mismatched_within[i]
            len_max = min(len(pairs_sets[i]), len(pairs_sets[j]))
            for k in range(len_max):
                di = list(pairs_sets[i])[k]
                dj = list(pairs_sets[j])[k]
                if di[0] in vocabulary and dj[1] in vocabulary and dj[1] != di[0]:
                    offsets_mismatched_within[-1][-1].append(
                        model.wv.get_vector(dj[1]) - model.wv.get_vector(di[0]))

    # a* - a, a et a* de categories différentes, probablement très très grand pour bats!!
    print("# Computing random offsets for mismatched across")
    offsets_mismatched_across = []
    perm_lists_mismatched_across = []
    for k_r in range(nb_random):
        perm_list_mismatched_across = permutation_onecycle(len(pairs_sets))
        perm_lists_mismatched_across.append(perm_list_mismatched_across)
        offsets_mismatched_across.append([])

        for i in range(len(pairs_sets)):
            offsets_mismatched_across[-1].append([])
            j = perm_list_mismatched_across[i]
            len_max = min(len(pairs_sets[i]), len(pairs_sets[j]))
            for k in range(len_max):
                di = list(pairs_sets[i])[k]
                dj = list(pairs_sets[j])[k]
                if di[0] in vocabulary and dj[1] in vocabulary and dj[1] != di[0]:
                    offsets_mismatched_across[-1][-1].append(model.wv.get_vector(dj[1]) - model.wv.get_vector(di[0]))

    print("# Computing the half random offsets")
    # For half random categories
    idx_random_categ = []
    for k_d in range(len(pairs_sets)):
        idx_random_categ.append([])
        for k in range(nb_random):
            rand_ints = np.random.choice(limit_word, size=len(pairs_sets[k_d]), replace=False)
            rand_vos = [vocabulary_list[r] for r in rand_ints if not vocabulary_list[r] in vocab_used]  # i?
            while len(rand_vos) < len(pairs_sets[k_d]):
                rand_int = int(np.random.choice(limit_word, size=1, replace=False))
                if not vocabulary_list[rand_int] in vocab_used and not vocabulary_list[rand_int] in rand_vos:
                    rand_vos.append(vocabulary_list[rand_int])
            idx_random_categ[-1].append(rand_vos)
    idx_random_categ = np.array(idx_random_categ)

    # a* - a, a d'un ensemble random
    offsets_random_start = np.array([[[model.wv.get_vector(pairs_sets[k_d][i][1]) - \
                                        model.wv.get_vector(idx_random_categ[k_d][k_r][i])
                                        for i in range(len(pairs_sets[k_d]))
                                        if pairs_sets[k_d][i][1] in vocabulary]
                                        for k_d in range(len(pairs_sets))] for k_r in range(nb_random)])

    # a* - a, a* d'un ensemble random
    offsets_random_end = np.array([[[model.wv.get_vector(idx_random_categ[k_d][k_r][i]) - \
                                        model.wv.get_vector(pairs_sets[k_d][i][0])
                                        for i in range(len(pairs_sets[k_d]))
                                        if pairs_sets[k_d][i][0] in vocabulary]
                                        for k_d in range(len(pairs_sets))] for k_r in range(nb_random)])

    print("# Computing the fully random offsets")
    # For random->random categories
    idx_random_full_start = [np.random.choice(limit_word, size=size_random_categ, replace=False) for k in range(nb_random)]
    idx_random_full_start = np.array(
        [[vocabulary_list[i] for i in idx_random_full_start[k] if not vocabulary_list[i] in vocab_used] for k in range(nb_random)])

    idx_random_full_end = []
    for k in range(nb_random):
        rand_ints = np.random.choice(limit_word, size=size_random_categ, replace=False)
        rand_vos = [vocabulary_list[r] for r in rand_ints if
                    not vocabulary_list[r] in vocab_used and not vocabulary_list[r] in idx_random_full_start[k]]
        while len(rand_vos) < 50:
            rand_int = int(np.random.choice(limit_word, size=1, replace=False))
            if not vocabulary_list[rand_int] in vocab_used and not vocabulary_list[rand_int] in rand_vos and not \
            vocabulary_list[rand_int] in idx_random_full_start[k]:
                rand_vos.append(vocabulary_list[rand_int])
        idx_random_full_end.append(rand_vos)
    idx_random_full_end = np.array(idx_random_full_end)

    # a* - a, a et a* d'ensembles random
    offsets_random_full = [
        np.array([[model.wv.get_vector(idx_random_full_end[k_r][i]) - model.wv.get_vector(idx_random_full_start[k_r][i])
                   for i in range(len(idx_random_full_start[k_r]))]
                  ]) for k_r in range(nb_random)]

    offsets_random = (offsets_permutation_within,
                      offsets_mismatched_within,
                      offsets_mismatched_across,
                      offsets_random_start,
                      offsets_random_end,
                      offsets_random_full)

    perm_lists = (perm_lists_permutation_within,
                  perm_lists_mismatched_within,
                  perm_lists_mismatched_across)

    idx_randoms = (idx_random_categ,
                   idx_random_full_start,
                   idx_random_full_end)

    return(offsets_random,
           perm_lists,
           idx_randoms)

def shuffled_offsets_random(model, pairs_sets, vocabulary, perm_lists, idx_randoms, nb_perms=50, nb_random=10):
    perm_lists_permutation_within, \
    perm_lists_mismatched_within, \
    perm_lists_mismatched_across = perm_lists

    idx_random_categ, \
    idx_random_full_start, \
    idx_random_full_end = idx_randoms

    print("# Computing shuffled offsets")

    # a* - a, a et a* de categories différentes, même grande catégorie pour bats probablement très très grand pour bats!!
    print("# Computing shuffled offsets for mismatched within")
    offsets_mismatched_within_shuffle = []
    for k_r in range(nb_random):
        offsets_mismatched_within_shuffle.append([])
        perm_list_intra = perm_lists_mismatched_within[k_r]
        for k in range(len(pairs_sets)):
            offsets_mismatched_within_shuffle[-1].append([])
            kj = perm_list_intra[k]
            len_max = min(len(pairs_sets[k]), len(pairs_sets[kj]))
            for perm in range(nb_perms):
                perm_list = permutation_onecycle(len_max)
                # perm_list = permutation_onecycle_avoidtrue(len_max, directions_tuples[kj])
                dirs = [model.wv.get_vector(pairs_sets[kj][perm_list[i]][1]) -
                        model.wv.get_vector(pairs_sets[k][i][0])
                        for i in range(len_max) if pairs_sets[kj][perm_list[i]][1] in vocabulary and pairs_sets[k][i][0] in vocabulary]
                offsets_mismatched_within_shuffle[-1][-1].append(dirs)

    print("# Computing shuffled offsets for mismatched across")
    offsets_mismatched_across_shuffle = []
    for k_r in range(nb_random):
        offsets_mismatched_across_shuffle.append([])
        perm_list_across = perm_lists_mismatched_across[k_r]
        for k in range(len(pairs_sets)):
            offsets_mismatched_across_shuffle[-1].append([])
            kj = perm_list_across[k]
            len_max = min(len(pairs_sets[k]), len(pairs_sets[kj]))
            for perm in range(nb_perms):
                perm_list = permutation_onecycle(len_max)
                # perm_list = permutation_onecycle_avoidtrue(len_max, directions_tuples[kj])
                dirs = [model.wv.get_vector(pairs_sets[kj][perm_list[i]][1]) -
                        model.wv.get_vector(pairs_sets[k][i][0])
                        for i in range(len_max) if pairs_sets[kj][perm_list[i]][1] in vocabulary and pairs_sets[k][i][0] in vocabulary]
                offsets_mismatched_across_shuffle[-1][-1].append(dirs)

    print("# Computing shuffled offsets for random start")
    offsets_random_start_shuffle = []
    for k_r in range(nb_random):
        offsets_random_start_shuffle.append([])
        for k in range(len(pairs_sets)):
            offsets_random_start_shuffle[-1].append([])
            len_max = min(len(pairs_sets[k]), len(idx_random_categ[k][k_r]))
            for perm in range(nb_perms):
                perm_list = permutation_onecycle(len_max)
                dirs = [model.wv.get_vector(pairs_sets[k][perm_list[i]][1]) -
                        model.wv.get_vector(idx_random_categ[k][k_r][i])
                        for i in range(len_max) if pairs_sets[k][perm_list[i]][1] in vocabulary and idx_random_categ[k][k_r][i] in vocabulary]
                offsets_random_start_shuffle[-1][-1].append(dirs)

    # a* - a, a d'un ensemble random, shuffle
    print("# Computing shuffled offsets for random end")
    offsets_random_end_shuffle = []
    for k_r in range(nb_random):
        offsets_random_end_shuffle.append([])
        for k in range(len(pairs_sets)):
            offsets_random_end_shuffle[-1].append([])
            len_max = min(len(pairs_sets[k]), len(idx_random_categ[k][k_r]))
            for perm in range(nb_perms):
                perm_list = permutation_onecycle(len_max)
                dirs = [model.wv.get_vector(idx_random_categ[k][k_r][perm_list[i]]) -
                        model.wv.get_vector(pairs_sets[k][i][0])
                        for i in range(len_max) if idx_random_categ[k][k_r][perm_list[i]] in vocabulary and pairs_sets[k][i][0] in vocabulary]
                offsets_random_end_shuffle[-1][-1].append(dirs)


############### A CHANGER PEUT ETRE
    print("# Computing shuffled offsets for fully random")
    offsets_random_full_shuffle = []
    for k_r in range(nb_random):
        offsets_random_full_shuffle.append([])
        offsets_random_full_shuffle[-1].append([])
        for perm in range(nb_perms):
            perm_list = permutation_onecycle(len(idx_random_full_start[k_r]))
            dirs = [model.wv.get_vector(idx_random_full_end[k_r][perm_list[i]]) -\
                    model.wv.get_vector(idx_random_full_start[k_r][i])
                    for i in range(len(idx_random_full_start[k_r])) if idx_random_full_end[k_r][perm_list[i]] in vocabulary and idx_random_full_start[k_r][i] in vocabulary]
            offsets_random_full_shuffle[-1][-1].append(dirs)

    #offsets_random_full_shuffle = [
    #    np.array([[model.wv.get_vector(idx_random_full_end[k_r][i]) - model.wv.get_vector(idx_random_full_start[k_r][i])
    #               for i in range(len(idx_random_full_start[k_r]))]
    #              ]) for k_r in range(nb_random)]

    #offsets_random_full_shuffle = [[shuffled_directions(model, idx_random[k_r], idx_random2[k_r])
    #                                         for perm in range(nb_perm)] #!!!!!!!!!!???????
    #                                        for k_r in range(nb_random)]

    offsets_random_shuffle = (offsets_mismatched_within_shuffle,
                      offsets_mismatched_across_shuffle,
                      offsets_random_start_shuffle,
                      offsets_random_end_shuffle,
                      offsets_random_full_shuffle)

    return(offsets_random_shuffle)


def similarities_random(offsets_random, pairs_sets, vocabulary, nb_random=10):
    offsets_permutation_within, \
    offsets_mismatched_within, \
    offsets_mismatched_across, \
    offsets_random_start, \
    offsets_random_end, \
    offsets_random_full = offsets_random

    print("# Computing similarities for permutation within")
    similarities_permutation_within = [
        similarite_offsets(offsets_permutation_within[k_r]) for k_r in range(nb_random)]
    print("# Computing similarities for mismatched within")
    similarities_mismatched_within = [
        similarite_offsets(offsets_mismatched_within[k_r]) for k_r in range(nb_random)]
    print("# Computing similarities for mismatched across")
    similarities_mismatched_across = [similarite_offsets(offsets_mismatched_across[k_r]) for k_r
                                              in range(nb_random)]
    print("# Computing similarities for random start")
    similarities_random_start = [similarite_offsets(offsets_random_start[k_r])
                                                for k_r in range(nb_random)]
    print("# Computing similarities for random end")
    similarities_random_end = [similarite_offsets(offsets_random_end[k_r])
                                                for k_r in range(nb_random)]
    print("# Computing similarities for fully random")
    similarities_random_full = [similarite_offsets(offsets_random_full[k_r]) for k_r in
                                                range(nb_random)]

    similarities_random_results = (similarities_permutation_within,
                           similarities_mismatched_within,
                           similarities_mismatched_across,
                           similarities_random_start,
                           similarities_random_end,
                           similarities_random_full)

    return(similarities_random_results)

def similarities_shuffle_random(offsets_random_shuffle, nb_random=10, nb_perms=50):
    offsets_mismatched_within_shuffle, \
    offsets_mismatched_across_shuffle, \
    offsets_random_start_shuffle, \
    offsets_random_end_shuffle, \
    offsets_random_full_shuffle = offsets_random_shuffle

    print("# Computing similarities for fully random shuffle")
    similarities_random_full_shuffle = [
        [similarite_offsets(np.array(offsets_random_full_shuffle[k_r])[:, perm]) for perm in range(nb_perms)]
        for k_r in range(nb_random)]
    print("# Computing similarities for mismatched within shuffle")
    similarities_mismatched_within_shuffle = [
        [similarite_offsets(np.array(offsets_mismatched_within_shuffle[k_r])[:, perm]) for
         perm in range(nb_perms)] for k_r in range(nb_random)]
    print("# Computing similarities for mismatched across shuffle")
    similarities_mismatched_across_shuffle = [
        [similarite_offsets(np.array(offsets_mismatched_across_shuffle[k_r])[:, perm]) for perm in
         range(nb_perms)] for k_r in range(nb_random)]
    print("# Computing similarities for random start shuffle")
    similarities_random_start_shuffle = [
        [similarite_offsets(np.array(offsets_random_start_shuffle[k_r])[:, perm]) for perm in
         range(nb_perms)] for k_r in range(nb_random)]
    print("# Computing similarities for random end shuffle")
    similarities_random_end_shuffle = [
        [similarite_offsets(np.array(offsets_random_end_shuffle[k_r])[:, perm]) for perm in
         range(nb_perms)] for k_r in range(nb_random)]

    similarities_random_shuffle_results = (similarities_mismatched_within_shuffle,
                                   similarities_mismatched_across_shuffle,
                                   similarities_random_start_shuffle,
                                   similarities_random_end_shuffle,
                                   similarities_random_full_shuffle)

    return(similarities_random_shuffle_results)


def ocs_pcs_random(similarities, similarities_shuffle, similarities_random_results, similarities_random_shuffle_results, nb_random=10, nb_perms=50):
    similarities_permutation_within, \
    similarities_mismatched_within, \
    similarities_mismatched_across, \
    similarities_random_start, \
    similarities_random_end, \
    similarities_random_full = similarities_random_results

    similarities_mismatched_within_shuffle, \
    similarities_mismatched_across_shuffle, \
    similarities_random_start_shuffle, \
    similarities_random_end_shuffle, \
    similarities_random_full_shuffle = similarities_random_shuffle_results

    print('# Computing the OCS and PCS metrics for all sets')

    ocs, pcs = OCS_PCS(nb_perms,
                       similarities,
                       similarities_shuffle)

    metrics_tmp = np.array([OCS_PCS(nb_perms,
                                    similarities_permutation_within[k_r],
                                    similarities_shuffle) for k_r in range(nb_random)])
    ocs_permutation_within, pcs_permutation_within = metrics_tmp[:, 0], metrics_tmp[:, 1]

    metrics_tmp = np.array([OCS_PCS(nb_perms,
                                    similarities_mismatched_within[k_r],
                                    similarities_mismatched_within_shuffle[k_r]) for k_r in range(nb_random)])
    ocs_mismatched_within, pcs_mismatched_within = metrics_tmp[:, 0], metrics_tmp[:, 1]

    metrics_tmp = np.array([OCS_PCS(nb_perms,
                                    similarities_mismatched_across[k_r],
                                    similarities_mismatched_across_shuffle[k_r]) for k_r in range(nb_random)])
    ocs_mismatched_across, pcs_mismatched_across = metrics_tmp[:, 0], metrics_tmp[:, 1]

    metrics_tmp = np.array([OCS_PCS(nb_perms,
                                    similarities_random_start[k_r],
                                    similarities_random_start_shuffle[k_r]) for k_r in range(nb_random)])
    ocs_random_start, pcs_random_start = metrics_tmp[:, 0], metrics_tmp[:, 1]

    metrics_tmp = np.array([OCS_PCS(nb_perms,
                                    similarities_random_end[k_r],
                                    similarities_random_end_shuffle[k_r]) for k_r in range(nb_random)])
    ocs_random_end, pcs_random_end = metrics_tmp[:, 0], metrics_tmp[:, 1]

    metrics_tmp = np.array([OCS_PCS(nb_perms,
                                    similarities_random_full[k_r],
                                    similarities_random_full_shuffle[k_r]) for k_r in range(nb_random)])
    ocs_random_full, pcs_random_full = metrics_tmp[:, 0], metrics_tmp[:, 1]

    ocs_all =  np.hstack([ocs,
               np.mean(ocs_permutation_within, axis=0),
               np.mean(ocs_mismatched_within, axis=0),
               np.mean(ocs_mismatched_across, axis=0),
               np.mean(ocs_random_start, axis=0),
               np.mean(ocs_random_end, axis=0),
               np.mean(ocs_random_full, axis=0)])

    pcs_all =  np.hstack([pcs,
               np.mean(pcs_permutation_within, axis=0),
               np.mean(pcs_mismatched_within, axis=0),
               np.mean(pcs_mismatched_across, axis=0),
               np.mean(pcs_random_start, axis=0),
               np.mean(pcs_random_end, axis=0),
               np.mean(pcs_random_full, axis=0)])

    for o in ocs_all:
        print(len(o))

    return(ocs_all, pcs_all)

def metrics_random_from_model(model, nb_perms=50, nb_random=10, size_random_categ=50, limit_word=10000):
    names, pairs_sets_set = bats_names_pairs(dir="BATS_3.0")
    pairs_sets = [list(pairs_sets_set[i]) for i in range(len(pairs_sets_set))]
    vocabulary = vocabulary_model(model)

    names_all = []
    for n in names:
        names_all.append("Normal: "+ str(n))
    for n in names:
        names_all.append("Permutation within: "+ str(n))
    for n in names:
        names_all.append("Mismatched within: "+ str(n))
    for n in names:
        names_all.append("Mismatched across: "+ str(n))
    for n in names:
        names_all.append("Random start: "+ str(n))
    for n in names:
        names_all.append("Random end: "+ str(n))
    names_all.append("Random full")

    normal_offsets, shf_offsets = normal_and_shuffled_offsets(model,
                                                              pairs_sets, nb_perms=nb_perms)

    offsets_random, \
    perm_lists, \
    idx_randoms = offsets_perms_random(model,
                                       pairs_sets,
                                       vocabulary,
                                       nb_random=nb_random, size_random_categ=size_random_categ, limit_word=limit_word)

    offsets_random_shuffle = shuffled_offsets_random(model,
                                                     pairs_sets,
                                                     vocabulary,
                                                     perm_lists,
                                                     idx_randoms,
                                                     nb_perms=nb_perms, nb_random=nb_random)

    similarities = similarite_offsets(normal_offsets)
    similarities_shuffle = [similarite_offsets(np.array(shf_offsets)[:, perm])
                            for perm in range(nb_perms)]

    similarities_random_results = similarities_random(offsets_random, pairs_sets, vocabulary, nb_random=nb_random)
    similarities_random_shuffle = similarities_shuffle_random(offsets_random_shuffle,
                                                              nb_random=nb_random, nb_perms=nb_perms)

    ocs_all, pcs_all = ocs_pcs_random(similarities,
                                      similarities_shuffle,
                                      similarities_random_results,
                                      similarities_random_shuffle,
                                      nb_random=nb_random, nb_perms=nb_perms)

    return (names_all, ocs_all, pcs_all)

def save_metrics_random(ocs_all, pcs_all, name, names_all, nb_perms, nb_random):
    if not exists('results'):
        print("# ", str('results'), "not found, creating dir.")
        mkdir('results')

    print(len(names_all))
    print(len(ocs_all))

    df_ocs = pd.DataFrame(np.array([names_all, ocs_all]).T, columns=np.array(["Categories", "OCS"]))
    df_pcs = pd.DataFrame(np.array([names_all, pcs_all]).T, columns=np.array(["Categories", "PCS"]))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    namepath = str(name) + '-' + str(nb_perms) + '_perms-' + str(nb_random) + '_random-' + str(timestr) + '.csv'

    df_ocs.to_csv('results/random_ocs-' + namepath, index=False)
    df_pcs.to_csv('results/random_pcs-' + namepath, index=False)

    print("# Successfully saved the random metrics to ocs/pcs-"+str(namepath))

if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) < 2:
        raise("# Please provide a model (all, name, or filename for a custom model)")

    name = sys.argv[1]

    if len(sys.argv) > 2:
        nb_perms = int(sys.argv[2])
    else:
        nb_perms = 50
    if len(sys.argv) > 3:
        nb_random = int(sys.argv[3])
    else:
        nb_random = 10

    if name == 'all':
        for name in MODELS:
            model = load_model(name)
            names_all, ocs_all, pcs_all = metrics_random_from_model(model,
                                                                    nb_perms=nb_perms,
                                                                    nb_random=nb_random,
                                                                    size_random_categ=50,
                                                                    limit_word=10000)
            print("# Sucessfully computed the random OCS and PCS metrics from", str(name))
            save_metrics_random(ocs_all, pcs_all, name, names_all, nb_perms, nb_random)

    else:
        model = load_model(name)

        names_all, ocs_all, pcs_all = metrics_random_from_model(model,
                                                                nb_perms=nb_perms,
                                                                nb_random=nb_random,
                                                                size_random_categ=50,
                                                                limit_word=10000)
        print("# Sucessfully computed the random OCS and PCS metrics from", str(name))
        save_metrics_random(ocs_all, pcs_all, name, names_all, nb_perms, nb_random)

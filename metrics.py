import numpy as np
import sklearn

import sys
import pandas as pd
import time

from os.path import exists
from os import mkdir

from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from models import clean_pairs, load_model, MODELS
from read_bats import bats_names_pairs

import torch
import json
import os

def token_embedding(tokenizer, model, word):
    tokenized_text = tokenizer.tokenize(word)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    embeds = np.array([model[i] for i in indexed_tokens])
    embed = np.mean(embeds, axis=0)
    return(embed)

def permutation_onecycle(n):
    if type(n) == tuple:
        n1, n2 = n[0], n[1]
    else:
        n1, n2 = 0, n
    l=np.random.permutation(range(n1, n2))
    for i in range(n1, n2):
        if i==l[i-n1]:
            j=np.random.randint(n1, n2)
            while j==l[j-n1]:
                j=np.random.randint(n1, n2)
            l[i-n1], l[j-n1] = l[j-n1], l[i-n1]
    return(l)

def permutation_onecycle_avoidtrue(n, real): #May be a more optimal way
    test = False
    perm = permutation_onecycle(n)
    for i_r in range(len(real)):
        if real[i_r][1] == real[perm[i_r]][1]:
            test = True
    while test:
        test = False
        perm = permutation_onecycle(n)
        for i_r in range(len(real)):
            if real[i_r][1] == real[perm[i_r]][1]:
                test = True
    return(perm)

def shuffled_directions(model, idx_start, idx_end):
    perm_list = permutation_onecycle(len(idx_start))
    dirs = np.array([[model.get_vector(idx_end[perm_list[i]]) - model.get_vector(idx_start[i])
                                          for i in range(len(idx_start))]])
    return(dirs)

def similarite_offsets(list_offsets):
    sim_offsets = []
    for i in range(len(list_offsets)):
        sim_offsets.append([])
        list_tuples = list(list_offsets[i])
        for j in range(len(list_tuples)):
            for k in range(j+1,len(list_tuples)):
                sim_offsets[-1].append(cos_sim([list_tuples[j]], [list_tuples[k]])[0][0])
    return(np.array(sim_offsets))

def OCS_PCS(nb_perm, similarities, similarities_shuffle):
    ocs, pcs = [], []
    print('# Computing the OCS and PCS metrics')
    for i in range(len(similarities)):
        pcs_list = []
        for perm in range(nb_perm):
            y_true = [1 for j in range(len(similarities[i]))]+[0 for j in range(len(similarities_shuffle[perm][i]))]
            y_scores = list(similarities[i])+list(similarities_shuffle[perm][i])
            auc_temp = sklearn.metrics.roc_auc_score(y_true,y_scores)
            pcs_list.append(auc_temp)
        pcs.append(np.mean(pcs_list))
        ocs.append(np.mean(similarities[i]))
    print('# Computed the OCS and PCS metrics')
    return(ocs, pcs)

def word_embedding(model, word):
    if type(model) == list:
        # BERT or GPT-2
        model, tokenizer = model
        embedding = token_embedding(tokenizer, model, word)
    else:
        # gensim based model
        embedding = model.get_vector(word)
    return(embedding)

def context_sentence(name):
    with open(os.path.join('BATS_3.0','context_sentences.json')) as json_file:
        context_sentences = json.load(json_file)
    return(context_sentences[name[:3]])

def sublist(liste, pattern):
    indx = -1
    for i in range(len(liste)):
        if liste[i] == pattern[0] and liste[i:i+len(pattern)] == pattern:
           indx = i
    return indx

def offset_contextual(model, tokenizer, model_name, name, w1, w2):
    context = context_sentence(name)
    c1, c2 = context

    sentence = ' '.join([c1, w1, c2, w2])
    if model_name == 'gpt-context':
        sentence = "[CLS] " + sentence + " [SEP]"
    else:
        w1 = " "+w1
        w2 = " "+w2

    tokenized_sentence = tokenizer.tokenize(sentence)
    tokenized_w1 = tokenizer.tokenize(w1)
    tokenized_w2 = tokenizer.tokenize(w2)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        if model_name == 'gpt-context':
            segments_ids = [1] * len(tokenized_sentence)
            segments_tensors = torch.tensor([segments_ids])
            outputs = model(tokens_tensor, segments_tensors)
        else:
            outputs = model(tokens_tensor)
        hidden_states = outputs[2]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    token_vecs = []
    for token in token_embeddings:
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        token_vecs.append(cat_vec)

    idx_w1 = sublist(tokenized_sentence, tokenized_w1)
    idx_w2 = sublist(tokenized_sentence, tokenized_w2)
    len_w1 = len(tokenized_w1)
    len_w2 = len(tokenized_w2)

    embd_w1 = torch.mean(torch.stack(token_vecs[idx_w1:idx_w1 + len_w1 + 1]), dim=0)
    embd_w2 = torch.mean(torch.stack(token_vecs[idx_w2:idx_w2 + len_w2 + 1]), dim=0)

    return(embd_w2 - embd_w1)


def offset(model, w1, w2, name):
    if type(model) == list and len(model) == 3:
        model, tokenizer, model_name = model
        return(offset_contextual(model, tokenizer, model_name, name, w1, w2))
    else:
        return(word_embedding(model, w2) - \
               word_embedding(model, w1))


def offsets(model, pairs_sets, names=None):
    return (np.array([[offset(model, i[0], i[1], names[k])
                       for i in pairs_sets[k]]
                       for k in range(len(pairs_sets))]))

def shuffled_offsets(model, pairs_sets, nb_perms=50, avoid_true=True, names=None):
    shf_offsets = []
    for k in range(len(pairs_sets)):
        shf_offsets.append([])
        for perm in range(nb_perms):
            if avoid_true:
                perm_list = permutation_onecycle_avoidtrue(len(pairs_sets[k]), pairs_sets[k])
            else:
                perm_list = permutation_onecycle(len(pairs_sets[k]))
            offs = [offset(model, pairs_sets[k][i][0], pairs_sets[k][perm_list[i]][1], names[k])
                    for i in range(len(pairs_sets[k]))]
            shf_offsets[-1].append(offs)
    return (shf_offsets)

def normal_and_shuffled_offsets(model, pairs_sets, nb_perms=50, names=None):
    print('# Computing the normal and shuffled offsets')
    pairs_sets = clean_pairs(model, pairs_sets)

    normal_offsets = offsets(model, pairs_sets, names=names)
    shf_offsets = shuffled_offsets(model, pairs_sets, nb_perms=nb_perms, names=names)
    print('# Computed the normal and shuffled offsets')
    return(normal_offsets, shf_offsets)


def metrics_from_model(model, nb_perms=50):
    names, pairs_sets = bats_names_pairs(dir="BATS_3.0")

    normal_offsets, shf_offsets = normal_and_shuffled_offsets(model, pairs_sets, nb_perms=nb_perms, names=names)

    print('# Computing the similarities of the normal and shuffled offsets')
    similarities = similarite_offsets(normal_offsets)
    similarities_shuffle = [similarite_offsets(np.array(shf_offsets)[:, perm])
                            for perm in range(nb_perms)]
    print('# Computed the similarities of the normal and shuffled offsets')

    ocs, pcs = OCS_PCS(nb_perms, similarities, similarities_shuffle)

    return (names, ocs, pcs)

def save_metrics(ocs, pcs, name, names, nb_perms):
    if not exists('results'):
        print("# ", str('results'), "not found, creating dir.")
        mkdir('results')

    df_ocs = pd.DataFrame(np.array([names, ocs]).T, columns=np.array(["Categories", "OCS"]))
    df_pcs = pd.DataFrame(np.array([names, pcs]).T, columns=np.array(["Categories", "PCS"]))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    namepath = str(name) + '-' + str(nb_perms) + '_perms-'  + str(timestr) + '.csv'

    df_ocs.to_csv('results/ocs-' + namepath, index=False)
    df_pcs.to_csv('results/pcs-' + namepath, index=False)

    print("# Successfully saved the metrics to results/ocs/pcs-"+str(namepath))

if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) < 2:
        raise("# Please provide a model (all, name, or filename for a custom model)")

    name = sys.argv[1]

    if len(sys.argv) > 2:
        nb_perms = int(sys.argv[2])
    else:
        nb_perms = 50

    if name == 'all':
        for name in MODELS:
            model = load_model(name)
            if name == 'bert-context' or name == 'gpt2-context':
                model.append(name)
            names, ocs, pcs = metrics_from_model(model, nb_perms=nb_perms)
            print("# Sucessfully computed the OCS and PCS metrics from", str(name))
            save_metrics(ocs, pcs, name, names, nb_perms)

    else:
        model = load_model(name)
        names, ocs, pcs = metrics_from_model(model, nb_perms=nb_perms)
        print("# Successfully computed the OCS and PCS metrics from", str(name))
        save_metrics(ocs, pcs, name, names, nb_perms)

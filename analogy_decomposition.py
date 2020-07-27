import numpy as np
import pandas as pd
import time
import sys
from read_bats import bats_names_pairs
from models import vocabulary_model, load_model

def start_end_words(model, pairs_sets, vocabulary):
    start_words = np.array(
        [[model.wv.get_vector(i[0]) for i in pairs_sets[k] if i[0] in vocabulary and i[1] in vocabulary]
         for k in range(len(pairs_sets))])

    end_words = np.array(
        [[model.wv.get_vector(i[1]) for i in pairs_sets[k] if i[0] in vocabulary and i[1] in vocabulary]
         for k in range(len(pairs_sets))])

    return(start_words, end_words)

def analogy_decomposition(start_words, end_words):
    b_bp = []
    oa_ob = []
    oa_b = []

    for i in range(len(start_words)):
        b_bp_categ = []
        oa_ob_categ = []
        oa_b_categ = []

        list_start_words = list(start_words[i])
        list_end_words = list(end_words[i])

        for j in range(len(list_start_words)):
            for k in range(len(list_start_words)):
                if j != k:
                    a, ap, b, bp = list_start_words[j], list_end_words[j], list_start_words[k], list_end_words[k]
                    o_a, o_b = ap - a, bp - a
                    analogy = b + o_a
                    norme_analogie_m1 = 1 / (np.linalg.norm(analogy))
                    norme_bp_m1 = 1 / (np.linalg.norm(bp))

                    b_bp_categ.append(b @ bp * norme_analogie_m1 * norme_bp_m1)
                    oa_ob_categ.append(o_a @ o_b * norme_analogie_m1 * norme_bp_m1)
                    oa_b_categ.append(o_a @ b * norme_analogie_m1 * norme_bp_m1)

        b_bp.append(np.mean(b_bp_categ))
        oa_ob.append(np.mean(oa_ob_categ))
        oa_b.append(np.mean(oa_b_categ))


    return([b_bp, oa_ob, oa_b])


def analogy_decomposition_reference(start_words, end_words):
    analogy_b = []
    oa_b = []
    oa2 = []

    for i in range(len(start_words)):
        analogy_b_categ = []
        oa_b_categ = []
        oa2_categ = []

        list_start_words = list(start_words[i])
        list_end_words = list(end_words[i])

        for j in range(len(list_start_words)):
            for k in range(len(list_start_words)):
                if j != k:
                    a, ap, b, bp = list_start_words[j], list_end_words[j], list_start_words[k], list_end_words[k]
                    o_a, o_b = ap - a, bp - a
                    analogy = b + o_a
                    norme_analogie_m1 = 1 / (np.linalg.norm(analogy))

                    analogy_b_categ.append((norme_analogie_m1 ** 2) * analogy @ b)
                    oa_b_categ.append((norme_analogie_m1 ** 2) * o_a @ b)
                    oa2_categ.append((norme_analogie_m1 ** 2) * o_a @ o_a)

        analogy_b.append(np.mean(analogy_b_categ))
        oa_b.append(np.mean(oa_b_categ))
        oa2.append(np.mean(oa2_categ))

    return ([analogy_b, oa_b, oa2])

def delta_sim(start_words, end_words):
    analogy_b = []
    oa_ob = []
    b_ob = []

    for i in range(len(start_words)):
        analogy_b_categ = []
        oa_ob_categ = []
        b_ob_categ = []

        list_start_words = list(start_words[i])
        list_end_words = list(end_words[i])

        for j in range(len(list_start_words)):
            for k in range(len(list_start_words)):
                if j != k:
                    a, ap, b, bp = list_start_words[j], list_end_words[j], list_start_words[k], list_end_words[k]
                    o_a, o_b = ap - a, bp - a
                    analogy = b + o_a
                    norme_analogie_m1 = 1 / (np.linalg.norm(analogy))
                    norme_bp_m1 = 1 / (np.linalg.norm(bp))
                    norme_b_m1 = 1 / (np.linalg.norm(b))

                    analogy_b_categ.append(norme_analogie_m1 * (norme_bp_m1 - norme_b_m1) * (b @ analogy))
                    oa_ob_categ.append(norme_analogie_m1 * norme_bp_m1 * o_a @ o_b)
                    b_ob_categ.append(norme_analogie_m1 * norme_bp_m1 * b @ o_b)

        analogy_b.append(np.mean(analogy_b_categ))
        oa_ob.append(np.mean(oa_ob_categ))
        b_ob.append(np.mean(b_ob_categ))

    return ([analogy_b, oa_ob, b_ob])

def all_decompositions(start_words, end_words):
    results_decompo = analogy_decomposition(start_words, end_words)
    results_decompo_ref = analogy_decomposition_reference(start_words, end_words)
    results_delta_sim = delta_sim(start_words, end_words)
    return([results_decompo, results_decompo_ref, results_delta_sim])

def decompo(model, pairs_sets, vocabulary, decomposition='all'):
    start_words, end_words = start_end_words(model, pairs_sets, vocabulary)
    if decomposition=='all':
        return(all_decompositions(start_words, end_words))
    if decomposition=='decomposition':
        return (analogy_decomposition(start_words, end_words))
    if decomposition=='decomposition_ref':
        return (analogy_decomposition_reference(start_words, end_words))
    if decomposition=='delta_sim':
        return (delta_sim(start_words, end_words))

def save_decompo(names, results, decomposition):
    if decomposition == 'decomposition': columns = ['Categories','b*b\'', 'o_a*o_b', 'o_a*b']
    if decomposition == 'decomposition_ref': columns = ['Categories','b*analogy', 'o_a*b', 'o_a^2']
    if decomposition == 'delta_sim': columns = ['Categories','b*analogy', 'o_a*o_b', 'o_a*b']

    df = pd.DataFrame([names,results], columns=columns)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    namepath = str(decomposition) + '-' + str(timestr) + '.csv'
    df.to_csv(namepath, index=False)
    print("# Successfully saved the decomposition to ", str(namepath))

if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) < 2:
        raise("# Please provide a model (name, or filename for a custom model)")
    if len(sys.argv) < 3:
        raise("# Please provide a decomposition type (decomposition, decomposition_ref, delta_sim or all)")

    model_name = str(sys.argv[1])
    decomposition = str(sys.argv[2])

    print("# Model: ", model_name, " ; Decomposition: ", decomposition)

    names, pairs_sets = bats_names_pairs(dir="BATS_3.0")
    print('# Loading model for decompositions')
    model = load_model(model_name)
    vocabulary = vocabulary_model(model)

    print('# Computing the decompositions')
    results = decompo(model, pairs_sets, vocabulary, decomposition=decomposition)

    print('# Saving the decompositions')
    if decomposition == 'all':
        save_decompo(names, results[0], 'decomposition')
        save_decompo(names, results[1], 'decomposition_ref')
        save_decompo(names, results[2], 'delta_sim')
    else:
        save_decompo(names, results, decomposition)

    print("# Successfully computed the wanted decomposition of the model ", model_name)

import numpy as np

def start_end_words(model, pairs_sets, vocabulary):
    start_words = np.array(
        [[model.wv.get_vector(i[0]) for i in pairs_sets[k] if i[0] in vocabulary and i[1] in vocabulary]
         for k in range(len(pairs_sets))])

    end_words = np.array(
        [[model.wv.get_vector(i[1]) for i in pairs_sets[k] if i[0] in vocabulary and i[1] in vocabulary]
         for k in range(len(pairs_sets))])

    return(start_words, end_words)

def analogy_decomposition(start_words, end_words):
    b_bp=[]
    oa_ob=[]
    oa_b=[]

    for i in range(len(start_words)):
        b_bp.append([])
        oa_ob.append([])
        oa_b.append([])

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

                    b_bp[-1].append(b @ bp * norme_analogie_m1 * norme_bp_m1)
                    oa_ob[-1].append(o_a @ o_b * norme_analogie_m1 * norme_bp_m1)
                    oa_b[-1].append(o_a @ b * norme_analogie_m1 * norme_bp_m1)

    return(b_bp, oa_ob, oa_b)


def analogy_decomposition_reference(start_words, end_words):
    analogy_b = []
    oa_b = []
    oa2 = []

    for i in range(len(start_words)):
        analogy_b.append([])
        oa_b.append([])
        oa2.append([])

        list_start_words = list(start_words[i])
        list_end_words = list(end_words[i])

        for j in range(len(list_start_words)):
            for k in range(len(list_start_words)):
                if j != k:
                    a, ap, b, bp = list_start_words[j], list_end_words[j], list_start_words[k], list_end_words[k]
                    o_a, o_b = ap - a, bp - a
                    analogy = b + o_a
                    norme_analogie_m1 = 1 / (np.linalg.norm(analogy))

                    analogy_b[-1].append((norme_analogie_m1 ** 2) * analogy @ b)
                    oa_b[-1].append((norme_analogie_m1 ** 2) * o_a @ b)
                    oa2[-1].append((norme_analogie_m1 ** 2) * o_a @ o_a)

    return (analogy_b, oa_b, oa2)

def delta_sim(start_words, end_words):
    analogy_b = []
    oa_ob = []
    b_ob = []

    for i in range(len(start_words)):
        analogy_b.append([])
        oa_ob.append([])
        b_ob.append([])

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

                    analogy_b[-1].append(norme_analogie_m1 * (norme_bp_m1 - norme_b_m1) * (b @ analogy))
                    oa_ob[-1].append(norme_analogie_m1 * norme_bp_m1 * o_a @ o_b)
                    b_ob[-1].append(norme_analogie_m1 * norme_bp_m1 * b @ o_b)

    return (analogy_b, oa_ob, b_ob)

def all_decompositions(start_words, end_words):
    results_decompo = analogy_decomposition(start_words, end_words)
    results_decompo_ref = analogy_decomposition_reference(start_words, end_words)
    results_delta_sim = delta_sim(start_words, end_words)
    return(results_decompo, results_decompo_ref, results_delta_sim)

def decompo(model, pairs_sets, vocabulary, decomposition='all'):
    start_words, end_words = start_end_words(model, pairs_sets, vocabulary)
    if decomposition=='all':
        return(all_decompositions(start_words, end_words))
    if decomposition=='decomposition':
        return (analogy_decomposition(start_words, end_words))
    if decomposition=='decomposition_ref':
        return (analogy_decomposition_reference(start_words, end_words))
    if decomposition=='delta_sim':
        return(delta_sim(start_words, end_words))


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

def plot_decomposition():
    c1 = '#FFBBB3'
    c2 = '#5F2EFF'
    c3 = '#00610F'
    idx_idel = [37, 32, 39, 38, 36, 33, 35, 31, 30, 34, 29, 23, 24, 20, 26, 27, 25, 28, 21, 22, 19, 10, 14, 15, 11, 18,
                17,
                12, 13, 16, 2, 1, 0, 8,
                9, 4, 5, 6, 3, 7]
    x = np.array([d[5:-1] for d in directions_names])[idx_idel]

    y1 = [np.mean(am1_dm1_c_d[i]) for i in idx_idel]
    y2 = [np.mean(am1_dm1_bma_d[i]) for i in idx_idel]
    y3 = [np.mean(am1_dm1_bma_c[i]) for i in idx_idel]
    y4 = np.array(y2) - np.array(y3)
    #   y4=[np.mean(am1_dm1_c_dmc[i]) for i in idx_idel] même chose sauf erreur
    fig = go.Figure(go.Bar(x=x, y=y1, name='$b\cdot b^*$', marker_color=c2))
    fig.add_trace(go.Bar(x=x, y=y3, name='$b\cdot o_a$', marker_color=c1))
    fig.add_trace(go.Bar(x=x, y=y4, name='$o_b\cdot o_a$', marker_color=c3))
    fig.update_layout(barmode='relative',
                      xaxis={'tickangle': -45, 'ticklen': 0.5},
                      font=dict(family="Times New Roman", size=16),
                      yaxis_title_text='Value in the analogy score')
    fig.show()

    z1 = [np.mean(am2_ana_bma[i]) for i in idx_idel]
    z2 = [np.mean(am2_ana_c[i]) for i in idx_idel]
    z3 = [np.mean(am2_c_bma[i]) for i in idx_idel]
    z4 = [np.mean(am2_bma_bma[i]) for i in idx_idel]
    fig = go.Figure(go.Bar(x=x, y=z2, name='$b\cdot (b+o_a)$', marker_color=c2))
    fig.add_trace(go.Bar(x=x, y=z3, name='$b\cdot o_a$', marker_color=c1))
    fig.add_trace(go.Bar(x=x, y=z4, name='$off_a\cdot o_a$', marker_color=c3))
    fig.update_layout(barmode='relative',
                      xaxis={'tickangle': -45, 'ticklen': 0.5},
                      font=dict(family="Times New Roman", size=16),
                      yaxis_title_text='Value in the reference analogy score')
    fig.show()

    y1 = [np.mean(am1_cd_f_dm1mcm1[i]) for i in idx_idel]  # range(len(x))]
    y2 = [np.mean(am1_dm1_bma_dmc[i]) for i in idx_idel]  # range(len(x))]
    y3 = [np.mean(am1_dm1_c_dmc[i]) for i in idx_idel]  # range(len(x))]
    fig = go.Figure(go.Bar(x=x, y=y1, name='$(1\!-\!\|b^*\|/\|b\|)\!\cdot\!(b\!+\!o_a)\!\cdot\!b$', marker_color=c1))
    fig.add_trace(go.Bar(x=x, y=y2, name='$o_a\!\cdot\!o_b$', marker_color=c3))
    fig.add_trace(go.Bar(x=x, y=y3, name='$b\!\cdot\!o_b$', marker_color=c2))
    fig.update_layout(barmode='relative',
                      xaxis={'tickangle': -45, 'tickwidth': 0.5},
                      font=dict(family="Times New Roman", size=16),
                      yaxis_title_text='$\Delta_{sim}$')
    fig.show()


def plot_random():
    # word2vec roc

    labels = ["Derivationnal", "Encyclopedic", "Inflectional", "Lexicographic"]
    intervals = [(20, 30), (10, 20), (30, 40), (0, 10)]
    x = (ocs, ocs_nnp, ocs_categ_categ_intra, ocs_categ_categ, ocs_random_normal, ocs_normal_random, ocs_random_random)
    y = (pcs, pcs_nnp, pcs_categ_categ_intra, pcs_categ_categ, pcs_random_normal, pcs_normal_random, pcs_random_random)

    x_n, x_nnp, x_categ_categ_intra, x_categ_categ, x_random_normal, x_normal_random, x_random_random = x
    x_nnp, x_categ_categ_intra, x_categ_categ = np.mean(x_nnp, axis=0), np.mean(x_categ_categ_intra, axis=0), np.mean(
        x_categ_categ, axis=0)
    x_random_normal, x_normal_random, x_random_random = np.mean(x_random_normal, axis=0), np.mean(x_normal_random,
                                                                                                  axis=0), np.mean(
        x_random_random, axis=0)
    # x_nnp, x_categ_categ_intra, x_categ_categ = iqr(x_nnp,axis=0), iqr(x_categ_categ_intra,axis=0), iqr(x_categ_categ,axis=0)
    # x_random_normal, x_normal_random, x_random_random = iqr(x_random_normal,axis=0), iqr(x_normal_random,axis=0), iqr(x_random_random,axis=0)

    y_n, y_nnp, y_categ_categ_intra, y_categ_categ, y_random_normal, y_normal_random, y_random_random = y
    y_nnp, y_categ_categ_intra, y_categ_categ = np.mean(y_nnp, axis=0), np.mean(y_categ_categ_intra, axis=0), np.mean(
        y_categ_categ, axis=0)
    y_random_normal, y_normal_random, y_random_random = np.mean(y_random_normal, axis=0), np.mean(y_normal_random,
                                                                                                  axis=0), np.mean(
        y_random_random, axis=0)

    # y_nnp, y_categ_categ_intra, y_categ_categ = iqr(y_nnp,axis=0), iqr(y_categ_categ_intra,axis=0), iqr(y_categ_categ,axis=0)
    # y_random_normal, y_normal_random, y_random_random = iqr(y_random_normal,axis=0), iqr(y_normal_random,axis=0), iqr(y_random_random,axis=0)

    def l_m(x):
        return ([np.mean(x)])

    for i in range(len(labels)):
        i1, i2 = intervals[i]
        # x_cc = x_categ_categ[:,i1:i2]
        # y_cc = x_categ_categ[:,i1:i2]
        # for l in perm_lists_inter:

        # for icc in perm_list_categ_categ:
        #    if perm_list_categ_categ[icc] in range(i1,i2):
        #        x_cc = np.hstack((x_cc, x_categ_categ[icc]))
        #        y_cc = np.hstack((y_cc, y_categ_categ[icc]))
        print(labels[i])
        print("Normal                ", l_m(x_n[i1:i2])[0], l_m(y_n[i1:i2])[0])
        print("Permuted within categ ", l_m(x_nnp[i1:i2])[0], l_m(y_nnp[i1:i2])[0])
        print("Mismatched within type", l_m(x_categ_categ_intra[i1:i2])[0], l_m(y_categ_categ_intra[i1:i2])[0])
        print("Mismatched across type", l_m(x_categ_categ[i1:i2])[0], l_m(y_categ_categ[i1:i2])[0])
        print("Random start          ", l_m(x_normal_random[i1:i2])[0], l_m(y_normal_random[i1:i2])[0])
        print("Random end            ", l_m(x_random_normal[i1:i2])[0], l_m(y_random_normal[i1:i2])[0])
    print("Random start and end  ", l_m(x_random_random)[0], l_m(y_random_random)[0])


def plot_metrics():
    # word2vec roc

    labels = ["Lexicographic", "Encyclopedic", "Derivationnal", "Inflectional"]
    intervals = [(0, 10), (10, 20), (20, 30), (30, 40)]
    x = (ocs, ocs_nnp, ocs_categ_categ_intra, ocs_categ_categ, ocs_random_normal, ocs_normal_random, ocs_random_random)
    y = (pcs, pcs_nnp, pcs_categ_categ_intra, pcs_categ_categ, pcs_random_normal, pcs_normal_random, pcs_random_random)

    x_n, x_nnp, x_categ_categ_intra, x_categ_categ, x_random_normal, x_normal_random, x_random_random = x
    x_random_normal, x_normal_random, x_random_random = np.mean(x_random_normal, axis=0), np.mean(x_normal_random,
                                                                                                  axis=0), np.mean(
        x_random_random, axis=0)
    y_n, y_nnp, y_categ_categ_intra, y_categ_categ, y_random_normal, y_normal_random, y_random_random = y
    y_random_normal, y_normal_random, y_random_random = np.mean(y_random_normal, axis=0), np.mean(y_normal_random,
                                                                                                  axis=0), np.mean(
        y_random_random, axis=0)

    def l_m(x):
        return ([np.mean(x)])

    for i in range(len(labels)):
        i1, i2 = intervals[i]
        x_cc = x_categ_categ[i1:i2]
        y_cc = x_categ_categ[i1:i2]
        for icc in perm_list_categ_categ:
            if perm_list_categ_categ[icc] in range(i1, i2):
                x_cc = np.hstack((x_cc, x_categ_categ[icc]))
                y_cc = np.hstack((y_cc, y_categ_categ[icc]))
        print(labels[i])
        print("Normal")
        print(l_m(x_n[i1:i2]))
        print(l_m(y_n[i1:i2]))
        print("Normal NNP")
        print(l_m(x_nnp[i1:i2]))
        print(l_m(y_nnp[i1:i2]))
        print("CC intra")
        print(l_m(x_categ_categ_intra[i1:i2]))
        print(l_m(y_categ_categ_intra[i1:i2]))
        print("CC inter")
        print(l_m(x_categ_categ[i1:i2]))
        print(l_m(y_categ_categ[i1:i2]))
        print("Random->Normal")
        print(l_m(x_normal_random[i1:i2]))
        print(l_m(y_normal_random[i1:i2]))
        print("Normal->Random")
        print(l_m(x_random_normal[i1:i2]))
        print(l_m(y_random_normal[i1:i2]))
    print("Random->Random")
    print(l_m(x_random_random))
    print(l_m(y_random_random))
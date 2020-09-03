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

import numpy as np
import pandas as pd
import sys
from os.path import join, exists
from os import mkdir


import tensorflow
import transformers

from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

from pytorch_pretrained_bert import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel

def plot_decomposition(results, type):
    c1 = '#FFBBB3'
    c2 = '#5F2EFF'
    c3 = '#00610F'
    # Alphabetical order of the BATS names.
    idx_idel = [37, 32, 39, 38, 36, 33, 35, 31, 30, 34,
                29, 23, 24, 20, 26, 27, 25, 28, 21, 22,
                19, 10, 14, 15, 11, 18, 17, 12, 13, 16,
                2,  1,  0,  8,  9,  4,  5,  6,  3,  7]


    names, r1, r2, r3 = results

    x = np.array([d[5:-1] for d in names])[idx_idel][1:]

    r1 = r1[idx_idel][1:]
    r2 = r2[idx_idel][1:]
    r3 = r3[idx_idel][1:]

    #y1 = [np.mean(am1_dm1_c_d[i]) for i in idx_idel]
    #y2 = [np.mean(am1_dm1_bma_d[i]) for i in idx_idel]
    #y3 = [np.mean(am1_dm1_bma_c[i]) for i in idx_idel]
    #y4 = np.array(y2) - np.array(y3)
    #   y4=[np.mean(am1_dm1_c_dmc[i]) for i in idx_idel] même chose sauf erreur

    if type == 'decomposition':
        fig = go.Figure(go.Bar(x=x, y=r1, name='$b\cdot b^*$', marker_color=c2)) #y1
        fig.add_trace(go.Bar(x=x, y=r3, name='$b\cdot o_a$', marker_color=c1)) #y3
        fig.add_trace(go.Bar(x=x, y=r2, name='$o_b\cdot o_a$', marker_color=c3)) #y4
        fig.update_layout(barmode='relative',
                          xaxis={'tickangle': -45, 'ticklen': 0.5},
                          font=dict(family="Times New Roman", size=16),
                          yaxis_title_text='Value in the analogy score')
        return(fig)

    if type == 'decomposition_ref':
        fig = go.Figure(go.Bar(x=x, y=r1, name='$b\cdot (b+o_a)$', marker_color=c2))
        fig.add_trace(go.Bar(x=x, y=r2, name='$b\cdot o_a$', marker_color=c1))
        fig.add_trace(go.Bar(x=x, y=r3, name='$o_a\cdot o_a$', marker_color=c3))
        fig.update_layout(barmode='relative',
                          xaxis={'tickangle': -45, 'ticklen': 0.5},
                          font=dict(family="Times New Roman", size=16),
                          yaxis_title_text='Value in the reference analogy score')

        return (fig)
    #z1 = [np.mean(am2_ana_bma[i]) for i in idx_idel]
    #z2 = [np.mean(am2_ana_c[i]) for i in idx_idel]
    #z3 = [np.mean(am2_c_bma[i]) for i in idx_idel]
    #z4 = [np.mean(am2_bma_bma[i]) for i in idx_idel]
    #fig = go.Figure(go.Bar(x=x, y=z2, name='$b\cdot (b+o_a)$', marker_color=c2))
    #fig.add_trace(go.Bar(x=x, y=z3, name='$b\cdot o_a$', marker_color=c1))
    #fig.add_trace(go.Bar(x=x, y=z4, name='$off_a\cdot o_a$', marker_color=c3))
    #fig.update_layout(barmode='relative',
    #                  xaxis={'tickangle': -45, 'ticklen': 0.5},
    #                  font=dict(family="Times New Roman", size=16),
    #                  yaxis_title_text='Value in the reference analogy score')
    #fig.show()

    if type == 'delta':
        fig = go.Figure(
            go.Bar(x=x, y=r1, name='$(1\!-\!\|b^*\|/\|b\|)\!\cdot\!(b\!+\!o_a)\!\cdot\!b$', marker_color=c1))
        fig.add_trace(go.Bar(x=x, y=r2, name='$o_a\!\cdot\!o_b$', marker_color=c3))
        fig.add_trace(go.Bar(x=x, y=r3, name='$b\!\cdot\!o_b$', marker_color=c2))
        fig.update_layout(barmode='relative',
                          xaxis={'tickangle': -45, 'tickwidth': 0.5},
                          font=dict(family="Times New Roman", size=16),
                          yaxis_title_text='$\Delta_{sim}$')

        return (fig)

    #y1 = [np.mean(am1_cd_f_dm1mcm1[i]) for i in idx_idel]  # range(len(x))]
    #y2 = [np.mean(am1_dm1_bma_dmc[i]) for i in idx_idel]  # range(len(x))]
    #y3 = [np.mean(am1_dm1_c_dmc[i]) for i in idx_idel]  # range(len(x))]
    #fig = go.Figure(go.Bar(x=x, y=y1, name='$(1\!-\!\|b^*\|/\|b\|)\!\cdot\!(b\!+\!o_a)\!\cdot\!b$', marker_color=c1))
    #fig.add_trace(go.Bar(x=x, y=y2, name='$o_a\!\cdot\!o_b$', marker_color=c3))
    #fig.add_trace(go.Bar(x=x, y=y3, name='$b\!\cdot\!o_b$', marker_color=c2))
    #fig.update_layout(barmode='relative',
    #                  xaxis={'tickangle': -45, 'tickwidth': 0.5},
    #                  font=dict(family="Times New Roman", size=16),
    #                  yaxis_title_text='$\Delta_{sim}$')
    #fig.show()


#versions with "all"?

def plot_metrics(results, type):
    # OCS, PCS (not random only?), Analogy test vanilla or not
    c1 = '#5F2EFF'
    # Alphabetical order of the BATS names.
    idx_idel = [37, 32, 39, 38, 36, 33, 35, 31, 30, 34,
                29, 23, 24, 20, 26, 27, 25, 28, 21, 22,
                19, 10, 14, 15, 11, 18, 17, 12, 13, 16,
                2,  1,  0,  8,  9,  4,  5,  6,  3,  7]

    if type == 'ocs': yaxis_title_text = 'Offset Concentration Score'
    if type == 'pcs': yaxis_title_text = 'Pairwise Consistency Score'
    if type == 'analogy': yaxis_title_text = 'Analogy test scores'
    if type == 'ocs': yaxis_title_text = 'Honnest analogy test scores'

    if type in ['ocs', 'pcs']:
        names, r1 = results
        print(len(names))
        print(len(r1))
        print(names)
    else:
        names, r1, _, _ = results
        r1 = [0.01 * r1[i] for i in range(len(r1)) if not i in ['11','22','33','44']]

    x = np.array([d[5:-1] for d in names])[idx_idel][1:]

    r1 = r1[idx_idel][1:]

    fig = go.Figure(go.Bar(x=x, y=r1, marker_color=c1))
    #fig.add_trace(go.Bar(x=x, y=r2, name='$b\cdot o_a$', marker_color=c1))
    #fig.add_trace(go.Bar(x=x, y=r3, name='$o_a\cdot o_a$', marker_color=c3))
    fig.update_layout(xaxis={'tickangle': -45, 'ticklen': 0.5},
                      font=dict(family="Times New Roman", size=16),
                      yaxis_title_text=yaxis_title_text)

    if type == 'pcs':
        fig.update_yaxes(range=[0.5, 1])


    return (fig)

def save_img(img, path, img_type):
    if not exists('figures'):
        print("# ", str('figures'), "not found, creating dir.")
        mkdir('figures')

    img.write_image("figures/" + str(path[:-4]) + '.' + img_type)

def plot_result(path):
    spath = str(path)
    if 'ocs' in spath or 'pcs' in spath or 'analogy_test' in spath:
        results = pd.read_csv(path)
        if 'ocs' in spath: type='ocs'
        if 'pcs' in spath: type='pcs'

        return(plot_metrics(results, type=type))

    if 'decomposition' in spath or 'delta' in spath:
        results = pd.read_csv(path)
        if 'decomposition' in spath: type='decomposition'
        if 'decomposition_ref' in spath: type='decomposition_ref'
        if 'delta' in spath: type='delta'

        return(plot_decomposition(results, type=type))

if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) == 1:
        raise("# Please provide a result file in /results")
    else:
        path = sys.argv[1]
        if not exists(join('results',path)):
            raise ("# File not found in " + str(join('results',path)))

        if len(sys.argv) > 2:
            img_type = str(sys.argv[2])
            if not img_type in ['png','pdf','svg','jpeg','webp']:
                raise ("# Image type argument not valid: " + img_type + ' should be one of these types: png, pdf, svg, jpeg, webp')
        else:
            img_type = 'png'

        img = plot_result(join('results',path))

        save_img(img, path, img_type)
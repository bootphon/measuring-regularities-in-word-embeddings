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

def plot_decomposition(results, type):
    c1 = '#FFBBB3'
    c2 = '#5F2EFF'
    c3 = '#00610F'
    idx_idel = [37, 32, 39, 38, 36, 33, 35, 31, 30, 34,
                29, 23, 24, 20, 26, 27, 25, 28, 21, 22,
                19, 10, 14, 15, 11, 18, 17, 12, 13, 16,
                2,  1,  0,  8,  9,  4,  5,  6,  3,  7]


    names, r1, r2, r3 = results

    x = np.array([d[5:-1] for d in names])[idx_idel]

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


#versions with "all"?

def plot_metrics(results, type):
    # OCS, PCS (not random only?), Analogy test vanilla or not

    return()

def plot_result(path):
    spath = str(path)
    if 'ocs' in spath or 'pcs' in spath or 'analogy_test' in spath:
        results = pd.read_csv(path)
        if 'ocs' in spath: type='ocs'
        if 'pcs' in spath: type='pcs'

        return(plot_metrics(results, type=type))

if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) == 1:
        raise("# Please provide a result file in /results")
    else:
        path = sys.argv[1]
        if not exists(join('results',path)):
            raise ("# File not found in " + str(join('results',path)))

        img = plot_result(join('results',path))

        save_img(img, path)
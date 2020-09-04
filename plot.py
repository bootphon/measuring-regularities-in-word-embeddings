import sys
from os import mkdir
from os.path import join, exists

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# If plotly does not save the image, try:
# conda install psutil
# conda install -c plotly plotly-orca
# Saving the image may be more complex on Linux

def plot_decomposition(results, type):
    c1 = '#5F2EFF'
    c2 = '#FFBBB3'
    c3 = '#00610F'
    # Alphabetical order of the BATS names.
    idx_idel = [37, 32, 39, 38, 36, 33, 35, 31, 30, 34,
                29, 23, 24, 20, 26, 27, 25, 28, 21, 22,
                19, 10, 14, 15, 11, 18, 17, 12, 13, 16,
                2,  1,  0,  8,  9,  4,  5,  6,  3,  7]

    results = np.array(results)
    names, r1, r2, r3 = results[:, 0], results[:, 1], results[:, 2], results[:, 3]

    x = np.array([d[5:-1] for d in names])[idx_idel][1:]

    r1 = r1[idx_idel][1:]
    r2 = r2[idx_idel][1:]
    r3 = r3[idx_idel][1:]

    if type == 'decomposition':
        fig = go.Figure(go.Bar(x=x, y=r1, name='$b\cdot b^*$', marker_color=c1))  # y1
        fig.add_trace(go.Bar(x=x, y=r3, name='$b\cdot o_a$', marker_color=c2))  #y3
        fig.add_trace(go.Bar(x=x, y=r2, name='$o_b\cdot o_a$', marker_color=c3)) #y4
        fig.update_layout(yaxis_title_text='Value in the analogy score')

    if type == 'decomposition_ref':
        fig = go.Figure(go.Bar(x=x, y=r1, name='$b\cdot (b+o_a)$', marker_color=c1))
        fig.add_trace(go.Bar(x=x, y=r2, name='$b\cdot o_a$', marker_color=c2))
        fig.add_trace(go.Bar(x=x, y=r3, name='$o_a\cdot o_a$', marker_color=c3))
        fig.update_layout(yaxis_title_text='Value in the reference analogy score')

    if type == 'delta':
        fig = go.Figure(
            go.Bar(x=x, y=r1, name='$(1\!-\!\|b^*\|/\|b\|)\!\cdot\!(b\!+\!o_a)\!\cdot\!b$', marker_color=c2))
        fig.add_trace(go.Bar(x=x, y=r2, name='$o_a\!\cdot\!o_b$', marker_color=c3))
        fig.add_trace(go.Bar(x=x, y=r3, name='$b\!\cdot\!o_b$', marker_color=c1))
        fig.update_layout(yaxis_title_text='$\Delta_{sim}$')

    fig.update_layout(barmode='relative',
                      xaxis={'tickangle': -45, 'ticklen': 0.5})
    return (fig)

def plot_metrics(results, type):
    c1 = '#5F2EFF'
    # Alphabetical order of the BATS names.
    idx_idel = [37, 32, 39, 38, 36, 33, 35, 31, 30, 34,
                29, 23, 24, 20, 26, 27, 25, 28, 21, 22,
                19, 10, 14, 15, 11, 18, 17, 12, 13, 16,
                2,  1,  0,  8,  9,  4,  5,  6,  3,  7]

    if type == 'ocs': yaxis_title_text = 'Offset Concentration Score'
    if type == 'pcs': yaxis_title_text = 'Pairwise Consistency Score'
    if type == 'test': yaxis_title_text = 'Analogy test scores'
    if type == 'vanilla': yaxis_title_text = 'Honnest analogy test scores'

    results = np.array(results)
    names, r1 = results[:, 0], results[:, 1]

    if type in ['test', 'vanilla']:
        r1 = np.array([0.01 * r1[i] for i in range(len(r1)) if not i in [10, 21, 32, 43]])
        names = np.array([names[i] for i in range(len(names)) if not i in [10, 21, 32, 43]])
        x = names[idx_idel][1:]
    else:
        x = np.array([d[5:-1] for d in names])[idx_idel][1:]

    r1 = r1[idx_idel][1:]

    fig = go.Figure(go.Bar(x=x, y=r1, marker_color=c1))
    fig.update_layout(xaxis={'tickangle': -45, 'ticklen': 0.5},
                      yaxis_title_text=yaxis_title_text)

    if type == 'pcs':
        fig.update_yaxes(range=[0.5, 1])

    return (fig)

def save_img(img, path, img_type):
    if not exists('figures'):
        print("# ", str('figures'), "not found, creating dir.")
        mkdir('figures')

    img_path = "figures/" + str(path[:-4]) + '.' + img_type
    img.write_image(img_path, width=1920 / 2, height=1080 / 2)
    print('# Successfully saved the image file to: ', img_path)

def plot_result(path):
    spath = str(path)
    if 'ocs' in spath or 'pcs' in spath or 'analogy_test' in spath:
        results = pd.read_csv(path)
        if 'ocs' in spath: type='ocs'
        if 'pcs' in spath: type='pcs'
        if 'analogy_test' in spath: type = 'test'
        if 'vanilla' in spath: type = 'vanilla'

        return(plot_metrics(results, type=type))

    if 'decomposition' in spath or 'delta' in spath:
        results = pd.read_csv(path)
        if 'decomposition' in spath: type='decomposition'
        if 'decomposition_ref' in spath: type='decomposition_ref'
        if 'delta' in spath: type='delta'

        return(plot_decomposition(results, type=type))

    raise ('# This result file type is not plotable.')

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
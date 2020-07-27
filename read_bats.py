import os
from gensim import utils
import numpy as np

def bats_names_pairs(dir="BATS_3.0"):
    names = []
    pairs_sets = []

    for d in os.listdir(dir):
        if d != 'metadata.json':
            for f in os.listdir(os.path.join(dir,str(d))):
                names.append(str(f)[:-4])
                pairs_sets.append(set())
                with utils.open_file(os.path.join(dir,str(d),str(f))) as fin:
                    for line_no, line in enumerate(fin):
                        line = utils.to_unicode(line)
                        a, b = [word.lower() for word in line.split()]
                        list_b = b.split('/')
                        if list_b[0] != a: #Keeping only the first analogy pair
                            pairs_sets[-1].add((a, list_b[0]))

    return (names, pairs_sets)

def vocab_bats(pairs_sets):
    vocab_bats = np.hstack(
        [np.hstack([[i[0] for i in pairs_sets[k]] for k in range(len(pairs_sets))]),
         np.hstack([[i[1] for i in pairs_sets[k]] for k in range(len(pairs_sets))])])

    return (vocab_bats)


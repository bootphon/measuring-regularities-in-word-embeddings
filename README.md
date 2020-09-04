# Measuring regularities in word embeddings

Implementation of the Python code used for the CoNLL 2020 article: "Analogies minus analogy test: measuring regularities in word embeddings".

This code allow easy computation of the Offset Concentration Score (OCS) and Pairwise Consistency Score (PCS) on a given model, pretrained or custom; on the Bigger Analogy Test Set dataset.

Other experiences of the paper can be replicated: 
- Decomposing the analogy score, the reference score, and $\Delta_sim$.
- Computing the OCS and PCS on randomized BATS sets.
- Computing the analogy test accuracy (for the normal and "honest" version) of a model.
- Plotting easily some results.


## Getting Started

### Prerequisites

The requirements of this repository are indicated in the requirements.txt file.

### Installing

After installing the required libraries, it is recommended to run:

```
python models.py
```

To download all necessary pretrained word embeddings for tests on your system: 
word2vec, Glove, dict2vec, ConceptNet Numberbatch, BERT and GPT-2 tokens.

## Computing the OCS and PCS metrics on a model

It is possible to compute the metrics either on one of the pretrained models, or a custom KeyedVectors model compatible with gensim.

### Example

The pretrained models list is: ['word2vec', 'glove', 'dict2vec', 'conceptnet', 'bert', 'gpt2'].
If the model is not in the list, it is considered a custom model name found in the "/models" directory.

nb_perms is the number of shuffled offset sets to compute the PCS on. Default value is 50.

```
python metrics.py model [nb_perms]
```

### Plot the result score

It is possible to plot OCS or PCS results by giving the csv result file in "/results" in parameter.

```
python plot.py pcs-model-X_perms-20200901-120000.csv                                                                   
```

## Experiments

The other possible experiments are presented here.

### Decomposition

Returns the decomposition terms of the three values studied in the paper: 'decomposition', 'decomposition_ref', 'delta_sim'; for the analgoy score, the reference analogy score, and finally $\Delta_sim$.
The type chosen is one of these three terms, or 'all'.

```
python analogy_decomposition.py model type
```

Possible to plot any of these three decompositions like before:

```
python plot.py decomposition-20200901-120000.csv                                                                   
```

### Arithmetic analogy test

Returns the accuracy, number of correctly predicted and total pairs, for the arithmetic analogy test for a model on the BATS dataset.
Both the standard and "honest" (allowing the prediction of input words) test results are returned. 
Possible to choose a specific model or 'all' for all the pretrained models.

```
python analogy_decomposition.py model
```

Possible to plot the test results for the normal and honest test separately:

```
python plot.py analogy_test[_vanilla]-20200901-120000.csv                                                                          
```

### Randomized BATS sets metrics

Returns the OCS and PCS for a model (or all pretrained); not only the BATS sets but all the randomized variants of BATS discussed in the papers.
nb_perms refer to the number of shuffled sets as before, default 50.
nb_random refer to the number of random sets are created before outputing the average metrics on these sets. Default value is 10.

```
python random_sets.py model [nb_perms] [nb_random]
```

## Authors

* Louis Fournier

## License

This project is licensed under the MIT License (?) - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* None of the libraries are models used are ours. We thank the original authors of these libraries and models.

# ReadMe
---

This is the source code for ICLR 2021 paper under review *Weakly Supervised Formula Learner for Solving Mathematical Problems* <br/>

# Environment

The code was written and tested on Python 3.8 and PyTorch 1.6.0. <br/>

The other required libraries including :

+ json

# Getting start

## Data preparation

We provide the program `regularize.py` for data pre-processing.

### Math23K
For `Math23K`, download `Math_23K.json` from `https://github.com/ShichaoSun/math_seq2tree` and put it into `./data/math23/`. Then run :

```
python ./data/math23/regularize.py
```

### MathQA
For `MathQA`, download `MathQA.zip` from `https://math-qa.github.io/`. Extract `train.json` and `test.json` to `./data/mathqa`. Then run :

```
python ./data/mathqa/regularize.py
```

## Training

We provide the prepared search logs generated in our experiments in `./saves/$dataset$/`. You can decide whether to use them for a quick validation or to restart the learning from scratch by modifying the `load_search_log` config in `config.py`. Other configurations such as the dataset to use and the number of batches for training can also be modified in `config.py`. To start the training, run :

```
python ./train.py
```

The whole learning process may take a long time. You can interrupt it at any time you like by a keyboard interruption, and the search log and model weights will be saved to `./saves/$dataset$/`. The learning can be restarted by assigning the `load_search_log` and `policynet_load_weight` in `config.py`.

## Testing

To test the model, assign the `policynet_load_weight` in `config.py`. Then run :

```
python ./test/py
```

## Using pre-trained word vectors

We provide the option to load the pre-trained word vectors. To activate this option, switch `policynet_load_word_embedding` to `True` in `config.py`.

### Math23K
For Math23K, download the 300d word vectors from `https://github.com/Embedding/Chinese-Word-Vectors`. Extract the raw `.txt` file to `./data/math23/`. Then run :

```
python ./data/math23/generate_weight.py
```

### MathQA
For MathQA, download the 300d word vectors from `https://nlp.stanford.edu/projects/glove/`. Extract the raw `.txt` file to `./data/mathqa/`. Then run :

```
python ./data/mathqa/generate_weight.py
```
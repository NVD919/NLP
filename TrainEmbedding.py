# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:53:46 2019

@author: Robot Hands | github.com/NVD919 | Discord: NVD#0729
"""
# =============================================================================
#  Script for training embeddings from a preprocessed CSV. 
# =============================================================================
from gensim.models import FastText as FT
from gensim.utils import tokenize
import time

# For keeping track of the time spent during training
training_time = time.time()

# Tokenizes and reads a corpus formatted as a CSV line by line. 
class csvIterator(object):
    def __iter__(self):
        path = ('processed messages.csv')
        with open(path) as fin:
            for line in fin:
                yield list(tokenize(line))

# Basic Hyperparameters for training an embedding. 
model = FT(size = 350, window = 5, min_count = 5)

# Builds a list of all words encountered while reading the corpus.
model.build_vocab(sentences = csvIterator())

# Sets the total number of words in the model to be equal to the number of words in the corpus. 
total_examples = model.corpus_count

# Trains the model. Epochs is an ML terms to refern to the number of times the model learns the training data.
model.train(sentences = csvIterator(), total_examples = total_examples, epochs = 5)

# Normalizes the vector length. Helpful for similarity comparisons later on.
model.init_sims(replace = True)

print('Time elapsed during training: {:.2f} minutes'.format((time.time() - training_time) / 60))

model.save('Embedding.model')

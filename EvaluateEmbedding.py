# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:06:32 2019

@author: Robot Hands | www.github.com/NVD919 | Discord: NVD#0729
"""

from gensim.test.utils import datapath
from gensim.models import FastText

model = FastText.load('Embedding.model')

# Compares the responses given by the model against a list of words and response pairs given by humans.
# Returns two tuples and a singlet:
# [(Pearson similarity), (Spearman rank coefficient), (Ratio of pairs w/ unknown words)]
similarities = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
print(similarities)

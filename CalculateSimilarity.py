# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:43:02 2020

@author: Robot Hands | www.github.com/NVD919 | NVD#0729
"""

from gensim.models import FastText
from flask import Flask, request
from nltk.corpus import stopwords

model = FastText.load('Embedding.model')

comparison = Flask(__name__)

@watcher.route('/', methods = ['POST'])
def distance():
    json = request.json
    print(json)
    sentence1 = json['sentence1'].lower().split()
    sentence2 = json['sentence2'].lower().split()
    stopword = stopwords.words("english")
    sentence1 = [w for w in sentence1 if w not in stopword]
    sentence2 = [w for w in sentence2 if w not in stopword]
    distance = model.wv.wmdistance(sentence1, sentence2)
    return f"{ {round(distance, 4)} }"

if __name__ == '__main__':
    comparison.run()

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:43:02 2020

@author: Robot Hands | www.github.com/NVD919 | NVD#0729

I've included a second method for querying the model in a commented out section below that doesn't require Flask.
"""

from gensim.models import FastText
from flask import Flask, request
from nltk.corpus import stopwords

model = FastText.load('Embedding.model')

comparison = Flask(__name__)

# You need to have Flask installed to be able to run this API.  Run it and post your sentences as JSON packets in Watchman or however you like to do that.
@comparison.route('/', methods = ['POST'])
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
    
"""
#Use this one if you just want to play around with it by feeding your own sentences in your IDE / Terminal

sentence1 = "I'm in quarantined at home."
sentence2 = "I can't go anywhere because of the virus."
stopword = stopwords.words("english")
sentence1 = [w for w in sentence1 if w not in stopword]
sentence2 = [w for w in sentence2 if w not in stopword]
distance = model.wv.wmdistance(sentence1, sentence2)
print("Numerical representation of the difference between sentences: {}".format(distance))
"""

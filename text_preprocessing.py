# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:44:46 2019

@author: Robot Hands | github.com/nvd919
"""
import time
import pandas as pd
import contractions
import re
import inflect
import unicodedata
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

filename = 'xxxx.zip'
def preprocessing(filename):
    
    # Usecols corresponds to the messages column inside the archive. 
    data = pd.read_csv(filename, compression = 'zip', index_col = None, header = None, usecols = [0], names = ['Messages'])
    
    #Starts a timer to keep track of the length of time for preprocessing
    start = time.time()
    
    def lowercase(words):
    #   Convert all characters to lowercase from list of tokenized words
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words
    
    def remove_punctuation(words):
    #   Remove punctuation from list of tokenized words
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words
    
    def replace_numbers(words):
    #   Replace all interger occurrences in list of tokenized words with textual representation
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words
    
    def remove_non_ascii(words):
    #   Remove non-ASCII characters from list of tokenized words
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words
    
    def lemmatize_verbs(words):
    #   Lemmatize verbs in into their base form
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas
    
    # =========================================================================
    #   The actual proprocessing stack.  The order is somewhat important, as
    #   the tokenizer will not work as intended when encountering contractions.
    #   The order for lowercasing and ascii / punctuation removal is not very
    #   important.
    # ========================================================================= 
    data["Messages"] = data["Messages"].apply(contractions.fix)
    data["Messages"] = data["Messages"].apply(word_tokenize)
    data["Messages"] = data["Messages"].apply(lowercase)
    data["Messages"] = data["Messages"].apply(remove_punctuation)
    data["Messages"] = data["Messages"].apply(remove_non_ascii)
    
    #   We use a filter in conjunction with a list of stopwords maintained by NLTK
    #   to catch stopwords.
    stop = stopwords.words('english')
    data["Messages"] = data["Messages"].apply(lambda x: list(filter(lambda y: y not in stop, x)))

    # =============================================================================
    #   Similarly, we count the occurrences of each word as they appear in
    #   messages and setup two different lists to to be able to check very
    #   common / rare words. We use the rare word filter to deal with spelling
    #   mistakes as we assume misspelled words will appear less frequently than
    #   an arbitrarily chosen threshold, determined by inspection. If the most
    #   frequently occurring words tend to be too memetic in nature, we have a
    #   filter to deal with that as well. 
    # =============================================================================
    word_counter = Counter([w for sublist in data["Messages"] for w in sublist])
    occurrences = word_counter.most_common()
    n = 10
    most_common = dict(occurrences[:n])
    least_common = {k:v for k, v in occurrences if v < 5}
    data["Messages"] = data["Messages"].apply(lambda x: list(filter(lambda y: y not in least_common, x)))  
#   data["Messages"] = data["Messages"].apply(lambda x: list(filter(lambda y: y not in most_common, x)))
   
    #   The final operation is to lemmatize verbs into their base form once
    #   they've made it past all the filters.
    data["Messages"] = data["Messages"].apply(lemmatize_verbs)
    
    print("\n{} most common words:\n".format(most_common))
    print("\n", data)
    print("\nPreprocessing elapsed time (seconds):", time.time() - start)
    return data

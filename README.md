# NLP
A Repo for NLP-type work

There's a saying in the Natural Language Processing community: "You will know a word by the company it keeps".

It refers to how NLP models learn languages; We train something called an embedding,
which is basically a gigantic compendium of words represented numerically as vectors.
What tends to happen is that words with similar meanings have vector representations similar to each other.
For example, words like car, automobile, and vehicle, should all have vectors somewhat similar to each other as they have
something to do with transportation.  On the other hand, those words should be not at all close to ones like hamburger, soda, or fries,
as those are to do with food.

To tie this back into the saying its a poetic way to interpret how NLP models learn and understand words, but also a spot on summary
of how we as humans learn and understand them as well.

With that out of the way, this project trains an embedding from a CSV of sentences, each on its own line, and all in one column. 

You run text_preprocessing.py first, because (as I understand it) languages are full of words that don't contribute a great deal
in terms of meaning (and, but, like, a, etc). Basically, it cleans up and chops words down to their stems.
Words like 'drove', 'driving', and 'driven' are all chopped down to their stem 'drive' for the purposes of learning.

TrainEmbedding does as the name implies and is the second step of the process.  

EvaluateEmbedding is for checking to see how it performs against a predetermined set of words and meanings as given by humans
that's built into the Gensim library.

CalculateSimilarity is a fun little demonstration of one of the things you can do with your NLP model.  It gives you a numerical
representation of the 'distance' between two input sentences.

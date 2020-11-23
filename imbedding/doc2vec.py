import pandas as pd
import numpy as np

texts = []
f = open('../preprocess/preprocessed_review.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts.append(oneline)

f = open('../preprocess/preprocessed_review2.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts.append(oneline)

print(len(texts))

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
# doc2vec2
#2 doc2vec_model = Doc2Vec(documents, vector_size=20, window=10, min_count=20, workers=4)
#3 doc2vec_model = Doc2Vec(documents, vector_size=50, window=7, min_count=20, workers=4)
#4 doc2vec_model = Doc2Vec(documents, vector_size=100, window=7, min_count=20, workers=4)

doc2vec_model = Doc2Vec(documents, vector_size=50, window=5, min_count=20, workers=4)
doc2vec_model.save('doc2vec7.model')
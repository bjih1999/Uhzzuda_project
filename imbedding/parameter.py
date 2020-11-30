import pandas as pd
import numpy as np

texts = []
f = open('preprocess/jull_review.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts.append(oneline)

print(len(texts))

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

doc2vec_model = Doc2Vec(documents, vector_size=100, window=10, min_count=30, workers=4)
doc2vec_model.save('doc2vec_v=100_dm0.model')

doc2vec_model = Doc2Vec(documents, vector_size=100, window=10, min_count=30, workers=4)
doc2vec_model.save('doc2vec_v=100_dm1.model')
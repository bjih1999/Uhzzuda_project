import pandas as pd
import numpy as np

texts = []
f = open('../preprocess/preprocessed_review_sent_jahee.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    if(len(oneline) > 1):
        texts.append(oneline)

# print(len(texts))

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

doc2vec_model = Doc2Vec(documents, vector_size=300, window=10, min_count=30, workers=4)
doc2vec_model.save('doc2vec_jaehee1.model')

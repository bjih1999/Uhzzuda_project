from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import numpy as np
import csv

preprocessed_texts = []
f = open('2_tunning_shuffled_pre_review_rhino_1203.csv', 'r')
for line in f.readlines():
    oneline = line.replace('숙,제', '숙제')
    oneline = oneline.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(preprocessed_texts)]

doc2vec_model = Doc2Vec(documents, vector_size=300, window=10, min_count=50, workers=4, negative=10)
doc2vec_model.save('./doc2vec/4.model')

'''
# model = doc2vec_model
model = Doc2Vec.load('./doc2vec/2.model')
print("================= new2 ===================")
print(model.most_similar('많', topn=10))
print(model.most_similar('적', topn=10))
'''
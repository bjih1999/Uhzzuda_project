from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import numpy as np
import csv

preprocessed_texts = []
cnt = 0
f = open('3_tunning_shuffled_pre_review_rhino_1203.csv', 'r')
for line in f.readlines():
    oneline = line.replace('숙,제', ',숙제,')
    oneline = oneline.replace('기,말', ',기말,')
    oneline = oneline.replace('기,출', ',기출,')
    oneline = oneline.replace('기,대', ',기대,')
    oneline = oneline.replace('기,초', ',기초,')
    oneline = oneline.replace('알,채', ',안,채')
    oneline = oneline.replace('안채,워', ',안,채우')
    oneline = oneline.replace('재수,강', ',재수강,')
    oneline = oneline.replace('싸,강', ',싸강,')
    oneline = oneline.replace('인,강', ',싸강,')
    oneline = oneline.replace('강,의', ',강의,')
    oneline = oneline.replace('빡,세', ',빡세,')
    oneline = oneline.replace('빡,쎄', ',빡세,')
    oneline = oneline.replace('빡,셉', ',빡세,')
    oneline = oneline.replace('빡,셉', ',빡세,')
    oneline = oneline.replace('빡,치', ',빡치,')
    oneline = oneline.replace('빡,침', ',빡치,')
    oneline = oneline.replace('외,우', ',외우,')
    oneline = oneline.replace('외,울', ',외우,')
    oneline = oneline.replace('외,운', ',외우,')
    oneline = oneline.replace('전범,위', ',전체,범위,')
    oneline = oneline.replace('저,범위', ',전체,범위,')
    oneline = oneline.replace('오프,', ',오프라인,')
    oneline = oneline.replace('오프라,', ',오프라인,')
    oneline = oneline.replace('온오,프', ',온오프,')
    oneline = oneline.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)

print(cnt)

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(preprocessed_texts)]

doc2vec_model = Doc2Vec(documents, vector_size=300, window=10, min_count=70, workers=4, negative=10)
doc2vec_model.save('./doc2vec/8.model')



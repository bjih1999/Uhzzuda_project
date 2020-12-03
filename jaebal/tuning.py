from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import numpy as np
import csv

preprocessed_texts = []
f = open('tunning_shuffled_pre_review_rhino_1203.csv', 'r')
for line in f.readlines():
    oneline = line.replace('숙,제', '숙제')
    oneline = oneline.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)

'''
texts = []
oneline = []
for line in preprocessed_texts:
    for word in line:
        if word == '꾸':
            word = '꿀'
        elif word == '굉장히' or word == '아주' or word == '되게' or word == '엄청':
            word = '매우'
        elif word == '성적':
            word = '학점'
        elif word == '레포트' or word == '보고서':
            word = '숙제'
        elif word == '조원':
            word = '팀원'
        oneline.append(word)
    texts.append(oneline)


count = 0
with open('tunning_shuffled_preprocessed_review_1203.csv', 'w', newline='') as f:
    makewrite = csv.writer(f)
    for row_prediction in prediction:
        for row in texts:
            makewrite.writerow(row)


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(preprocessed_texts)]

doc2vec_model = Doc2Vec(documents, vector_size=300, window=10, min_count=50, workers=4)
# doc2vec_model.save('./doc2vec/1.model')
'''

print("======학점 숙제 시험 교수 강의======")
# model = doc2vec_model
model = Doc2Vec.load('./doc2vec/1.model')
print("================= new1 ===================")
print(model.most_similar('많', topn=10))
print(model.most_similar('적', topn=10))
from gensim.models.doc2vec import Doc2Vec
from scipy import spatial
import pandas as pd
import numpy as np

texts = []
f = open('./preprocessed_review_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    if(len(oneline) > 1):
        texts.append(oneline)


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

print("======학점 숙제 시험 교수 강의======")

model = Doc2Vec.load('./1.model')
print("================= 1 ===================")
print(model.most_similar('학점', topn=20))
print(model.most_similar('숙제', topn=20))
print(model.most_similar('시험', topn=20))
print(model.most_similar('교수', topn=20))
print(model.most_similar('강의', topn=20))

model = Doc2Vec.load('./2.model')
print("================= 2 ===================")
print(model.most_similar('학점', topn=20))
print(model.most_similar('숙제', topn=20))
print(model.most_similar('시험', topn=20))
print(model.most_similar('교수', topn=20))
print(model.most_similar('강의', topn=20))

model = Doc2Vec.load('./3.model')
print("================= 3 ===================")
print(model.most_similar('학점', topn=20))
print(model.most_similar('숙제', topn=20))
print(model.most_similar('시험', topn=20))
print(model.most_similar('교수', topn=20))
print(model.most_similar('강의', topn=20))

model = Doc2Vec.load('./4.model')
print("================= 4 ===================")
print(model.most_similar('학점', topn=20))
print(model.most_similar('숙제', topn=20))
print(model.most_similar('시험', topn=20))
print(model.most_similar('교수', topn=20))
print(model.most_similar('강의', topn=20))
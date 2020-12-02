from gensim.models.doc2vec import Doc2Vec
from scipy import spatial
import pandas as pd
import numpy as np

texts = []
f = open('../preprocess/preprocessed_review_jeol.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    if(len(oneline) > 1):
        texts.append(oneline)
'''
print("\n\n======= 모델1 =========")
model = Doc2Vec.load('./doc2vec_jeol1.model')
print(model.most_similar('교수', topn=20))

print("\n\n======= 모델2 =========")
model = Doc2Vec.load('./doc2vec_jeol2.model')
print(model.most_similar('교수', topn=20))

print("\n\n======= 모델3 =========")
model = Doc2Vec.load('./doc2vec_jeol3.model')
print(model.most_similar('교수', topn=20))

print("\n\n======= 모델4 =========")
model = Doc2Vec.load('./doc2vec_jeol4.model')
print(model.most_similar('교수', topn=20))
'''
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

print("\n\n======= 모델6 =========")
model = Doc2Vec.load('./doc2vec_jeol6.model')

print(model.most_similar('쉽', topn=10))
print(model.most_similar('어려움', topn=10))
print(model.most_similar('어렵', topn=10))
print("\n\n======= 시험이어렵다 =========")
string1 = ['시험', '어렵', '어려움']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)

print("\n\n======= 시험이쉽다=========")
string1 = ['시험', '쉽']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)


'''
print(model.most_similar('학점', topn=10))
print(model.most_similar('과제', topn=10))
print(model.most_similar('시험', topn=10))
print(model.most_similar('교수님', topn=10))
print(model.most_similar('강의력', topn=10))
print(model.most_similar('강의', topn=10))
print(model.most_similar('친절', topn=10))
print(model.most_similar('가르치', topn=10))


print("\n\n======= 과제가많다. =========")
string1 = ['과제', '많']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)

print("\n\n======= 강의력이좋다. =========")
string1 = ['강의력', '좋']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)

print("\n\n======= 시험이어렵다 =========")
string1 = ['시험', '어렵']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)


print("\n\n======= 교수님이 친절하시다 =========")
string1 = ['교수님', '친절']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)



print("\n\n======= 모델2 =========")
model = Doc2Vec.load('./doc2vec_sent5_new.model')
print(model.most_similar('과제', topn=10))
print(model.most_similar('시험', topn=10))
print(model.most_similar('교수', topn=10))
print(model.most_similar('강의력', topn=10))
print(model.most_similar('성적', topn=10))
print(model.most_similar('많', topn=10))

print("\n\n======= 과제가많다. =========")
string1 = ['과제', '많']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)

print("\n\n======= 강의력이좋다. =========")
string1 = ['강의력', '좋']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)

print("\n\n======= 시험이어렵다 =========")
string1 = ['시험', '어렵']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs

print("\n\n======= 모델3 =========")
model = Doc2Vec.load('./doc2vec_sent3.model')
print(model.most_similar('과제', topn=10))
print(model.most_similar('시험', topn=10))
print(model.most_similar('교수', topn=10))
print(model.most_similar('강의력', topn=10))
print(model.most_similar('성적', topn=10))
print(model.most_similar('많', topn=10))

print("\n\n======= 시험이어렵다 =========")
string1 = ['시험', '어렵']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)

print("\n\n======= 모델4 =========")
model = Doc2Vec.load('./doc2vec_sent4.model')
print(model.most_similar('과제', topn=10))
print(model.most_similar('시험', topn=10))
print(model.most_similar('교수', topn=10))
print(model.most_similar('강의력', topn=10))
print(model.most_similar('성적', topn=10))
print(model.most_similar('많', topn=10))

print("\n\n======= 시험이어렵다 =========")
string1 = ['시험', '어렵']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)


#string1 = "과제가 많다"
#vec1 = model.infer_vector(string1.split())
print("\n\n======= 과제가많다. =========")
from doc2vec2 import texts
string1 = ['과제', '많']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)

print("\n\n======= 강의력이좋다. =========")
string1 = ['강의력', '좋']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)


print("\n\n======= 시험이어렵다 =========")
string1 = ['시험', '어렵']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)
'''
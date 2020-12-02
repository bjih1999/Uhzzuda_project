from gensim.models.doc2vec import Doc2Vec
from scipy import spatial
import pandas as pd
import numpy as np

texts = []
f = open('../preprocess/preprocessed_review_sent_jahee.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    if(len(oneline) > 1):
        texts.append(oneline)


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

model = Doc2Vec.load('./model/doc2vec_jaehee1.model')
print(model.most_similar('성적', topn=20))
print(model.most_similar('점수', topn=20))
print(model.most_similar('학점', topn=20))
'''
print("\n\n======= 과제가 매주. =========")
string1 = ['과제', '매주']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    # print(most_similar_docs)

print("\n\n======= 교수님. 강의력. =========")
string1 = ['교수님', '좋', '최고']
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

from sklearn.cluster import KMeans
print("##"*30)
print("K-Means Clustering")
print("##"*30)
Clustering_Method = KMeans(n_clusters=7, random_state=0)
X = model.docvecs.vectors_docs # document vector 전체를 가져옴.
Clustering_Method.fit(X)# fitting
# 결과를 보면 알겠지만, 생각보다 클러스터링이 잘 되지 않음.
# 일단은 이것 또한 트레이닝 셋이 적어서 그런 것으로 보임.
cluster_dict = {i:[] for i in range(0, 3)}
for text_tags, label in zip(documents, Clustering_Method.labels_):
    text, tags = text_tags
    cluster_dict[label].append(text)
for label, lst in cluster_dict.items():
    print(f"Cluster {label}")
    for x in lst:
        print(x)
    print("--"*30)
print("##"*20)

print(model.most_similar('학점', topn=10))
print(model.most_similar('과제', topn=10))
print(model.most_similar('시험', topn=10))
print(model.most_similar('교수님', topn=10))
print(model.most_similar('강의력', topn=10))
print(model.most_similar('강의', topn=10))
print(model.most_similar('친절', topn=10))
print(model.most_similar('가르치', topn=10))
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
'''
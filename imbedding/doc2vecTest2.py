from gensim.models.doc2vec import Doc2Vec
from scipy import spatial

print("\n\n======= 모델1 =========")
model = Doc2Vec.load('./doc2vec_jull.model')
print(model.most_similar('과제', topn=10))
print(model.most_similar('시험', topn=10))
print(model.most_similar('교수', topn=10))
print(model.most_similar('강의력', topn=10))
print(model.most_similar('성적', topn=10))
print(model.most_similar('많', topn=10))

print("\n\n======= 모델2 =========")
model = Doc2Vec.load('./doc2vec_jull2.model')
print(model.most_similar('과제', topn=10))
print(model.most_similar('시험', topn=10))
print(model.most_similar('교수', topn=10))
print(model.most_similar('강의력', topn=10))
print(model.most_similar('성적', topn=10))
print(model.most_similar('많', topn=10))

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
from gensim.models.doc2vec import Doc2Vec
from scipy import spatial

#string1 = "과제가 많다"
#vec1 = model.infer_vector(string1.split())
model = Doc2Vec.load('imbedding/doc2vec_v100_w10_dm1.model');
print("\n\n======= 과제가많다. =========")
#from .doc2vec2 import texts
string1 = ['과제', '많']
vec1 = model.infer_vector(string1)
most_similar_docs = model.docvecs.most_similar([vec1], topn=10)
for index, similarity in most_similar_docs:
    print(f"{index}, similarity: {similarity}")
    print(texts[index])
    print(most_similar_docs)

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

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

doc2vec_model = Doc2Vec(documents, vector_size=300, window=10, min_count=50, workers=4)
doc2vec_model.save('./doc2vec/1.model')


print("======학점 숙제 시험 교수 강의======")
#model = doc2vec_model
model = Doc2Vec.load('./doc2vec/1.model')
print("================= new1 ===================")
print(model.most_similar('조', topn=30))
print(model.most_similar('별', topn=30))
print(model.most_similar('팀', topn=30))
print(model.most_similar('프로젝트', topn=30))
print(model.most_similar('매우', topn=30))


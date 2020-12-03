
from gensim.models.doc2vec import Doc2Vec
print("======학점 숙제 시험 교수 강의======")
#model = doc2vec_model
model = Doc2Vec.load('./doc2vec/4.model')
print("================= new1 ===================")
print(model.most_similar('조', topn=30))
print(model.most_similar('별', topn=30))
print(model.most_similar('팀', topn=30))
print(model.most_similar('프로젝트', topn=30))
print(model.most_similar('학점', topn=30))



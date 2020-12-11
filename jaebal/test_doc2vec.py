
from gensim.models.doc2vec import Doc2Vec
print("======학점 숙제 시험 교수 강의======")
model = Doc2Vec.load('./doc2vec/3.model')
print("================= new1 ===================")
print(model.most_similar('강', topn=30))
print(model.most_similar('싸', topn=30))
print(model.most_similar('강의', topn=30))


#model = doc2vec_model
model = Doc2Vec.load('./doc2vec/9.model')
print("================= new1 ===================")
print(model.most_similar('강', topn=30))
print(model.most_similar('싸강', topn=30))
print(model.most_similar('강의', topn=30))
# print(model.most_similar(positive=['학점'], negative=['F']))


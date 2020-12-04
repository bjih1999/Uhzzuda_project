
from gensim.models.doc2vec import Doc2Vec
print("======학점 숙제 시험 교수 강의======")
#model = doc2vec_model
model = Doc2Vec.load('./doc2vec/8.model')
print("================= new1 ===================")
print(model.most_similar('빡세', topn=30))
print(model.most_similar('강', topn=30))
print(model.most_similar('싸강', topn=30))
print(model.most_similar('범위', topn=30))
print(model.most_similar('오프라인', topn=30))
# print(model.most_similar(positive=['학점'], negative=['F']))


from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load('./doc2vec2.model')

print("K-Means Clustering")
print("##"*10)
clf = KMeans(n_clusters=7, random_state=0)
X = model.docvecs.vectors_docs # document vector 전체를 가져옴.
clf = clf.fit(X)
predict = pd.DataFrame(clf.predict(X))

predict.columns=['predict']


plt.scatter(r['Sepal length'],r['Sepal width'],c=r['predict'],alpha=0.5)


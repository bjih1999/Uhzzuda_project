from gensim.models.doc2vec import Doc2Vec
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv

num = 5000
preprocessed_texts = []
f = open('./shuffled_preprocessed_review_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)
    if len(preprocessed_texts) == num:
        break;


embedded_texts = []
model = Doc2Vec.load('../rhino/3.model')
for i in preprocessed_texts:
    embedded_texts.append(model.infer_vector(i))

size = (len(preprocessed_texts), 13)
labels = np.zeros(size, dtype=int)
print(len(labels))

f = open('label_5000.csv', 'r', encoding='UTF-8')
cnt = 0
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    for i in range(13):
        if oneline[i] == '1':
            labels[cnt][i] = 1
    cnt = cnt + 1
    if cnt == num:
        break

X = pd.DataFrame(embedded_texts)
y = pd.DataFrame(labels)

df_data = X.iloc[:, 0:300]
df_labels = y.iloc[:, 0:13]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.125, random_state=0)

'''
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
# param = {"hidden_layer_sizes": [(1,),(50,)], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005,0.0005], "max_iter": [4000]}
# clf = GridSearchCV(MLPClassifier(), params, cv=10, n_jobs=3)
clf = MLPClassifier(solver='adam', random_state=1, alpha=0.05, max_iter=1000)
clf.fit(X_train, y_train)
clf.fit(X_train, y_train)

print("train score : {}".format(clf.score(X_train, y_train)))
print("val score : {}".format(clf.score(X_test, y_test)))

# 모델 저장
import pickle
with open('mlp_model50_1.pkl', 'wb') as model_file:
  pickle.dump(clf, model_file)
print("저장완료")
'''
# 모델 불러오기
import pickle
clf = pickle.load(open('mlp_model50_1.pkl', 'rb'))

# 문장 분류하기
##원본 문장
review_texts = []
f = open('./shuffled_review_sent_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    review_texts.append(oneline)

wtf = []
for i in preprocessed_texts:
    wtf.append(model.infer_vector(i))

prediction = clf.predict(wtf)

# 6항목을 정답으로 분류한 결과 문장 (6은 교수님+)
count = 0
with open('mlp.csv', 'w', newline='') as f:
    makewrite = csv.writer(f)
    for row_prediction in prediction:
        for i in range(13):
            if row_prediction[i] == 1:
                makewrite.writerow(review_texts[count])
            else:
                makewrite.writerow('')
        count = count + 1

import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import csv

num = 5000
texts = []
f = open('./shuffled_preprocessed_review_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts.append(oneline)
    if len(texts) == num:
        break;

print(len(texts))

dataset = []
model = Doc2Vec.load('../rhino/3.model')
for i in texts:
    dataset.append(model.infer_vector(i))

s = (len(texts), 13)
labels = np.zeros(s, dtype=int)
print(len(labels))

ff = open('label_5000.csv', 'r', encoding='UTF-8')
cnt = 0
for line2 in ff.readlines():
    oneline = line2.replace("\n", "").split(",")
    for i in range(13):
        if oneline[i] == '1':
            labels[cnt][i] = 1
    cnt = cnt + 1
    if cnt == num:
        break


df = pd.DataFrame(dataset)
X = df.iloc[:, 0:300] # 벡터사이즈 지정 (300)


lf = pd.DataFrame(labels)
y = lf.iloc[:, 0:13]
# print('y : ', y)

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

print("train score : {}".format(clf.score(X_train, y_train)))
print("val score : {}".format(clf.score(X_test, y_test)))


# 모델 저장
import pickle
with open('mlp_model50_1.pkl', 'wb') as model_file:
  pickle.dump(clf, model_file)
print("저장완료")

'''
import pickle
clf = pickle.load(open('mlp_model50_1.pkl', 'rb'))


# 문장 분류하기
prediction = clf.predict(wtf)
print('prediction! \n', prediction)


with open('test_5000_1.csv','w', newline='') as f:
    makewrite = csv.writer(f)
    for value in prediction:
        makewrite.writerow(value)

texts2 = []
f = open('./shuffled_review_sent_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts2.append(oneline)
    if len(texts2) == num:
        break;

count = 0
with open('MLP_professor+.csv', 'w', newline='') as ffff:
    makewrite = csv.writer(ffff)
    for row_prediction in prediction:
        if row_prediction[6] == 1:
            makewrite.writerow(texts2[count])
        count = count + 1
'''
prms = clf.cv_results_['params']
acc_means = clf.cv_results_['mean_test_score']
acc_stds = clf.cv_results_['std_test_score']
for mean, std, prm in zip(acc_means, acc_stds, prms):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, prm))

import pickle
with open('mlp_model23_3.pkl', 'wb') as model_file:
  pickle.dump(mlp_multi, model_file)
print("저장완료")
'''

'''
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# 모델 불러오기
loaded_model = pickle.load(open('mlp_model23.pkl', 'rb'))

# Kfold
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
results = cross_val_score(loaded_model, X, y, cv=kfold)
print(results)
'''



'''
# 문장 분류하기
texts2 = []

f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts2.append(oneline)
    if len(testts2) == 1500:
        break

wtf = []
for i in texts2:
    wtf.append(model.infer_vector(i))

prediction = decision_tree.predict(wtf)
'''

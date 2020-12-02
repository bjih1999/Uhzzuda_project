import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import csv

texts = []
f = open('../rhino/review_sent_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts.append(oneline)
print(len(texts))

dataset = []
model = Doc2Vec.load('../rhino/1.model')
for i in texts:
    dataset.append(model.infer_vector(i))

s = (len(texts), 3)
labels = np.zeros(s, dtype=int)
print(len(labels))

ff = open('label_1500.csv', 'r', encoding='UTF-8')
cnt = 0
for line2 in ff.readlines():
    oneline = line2.replace("\n", "").split(",")
    # print(oneline[0],oneline[1], oneline[2])
    if oneline[0] == '1':
        labels[cnt][0] = 1
    if oneline[1] == '1':
        labels[cnt][1] = 1
    if oneline[2] == '1':
        labels[cnt][2] = 1
    cnt = cnt + 1

df = pd.DataFrame(dataset)
X = df.iloc[:, 0:300] # 벡터사이즈 지정 (300)

# 1500 개
lf = pd.DataFrame(labels)
y = lf.iloc[:, 0:13]
print('y : ', y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.125, random_state=0)

decision_tree = DecisionTreeClassifier(
    max_depth=1,
    # criterion='entropy',
    min_samples_split=20,
    # random_state=0,
    # min_samples_split=2,
    # min_samples_leaf=5,
    # max_features=None,
    # max_leaf_nodes=None,
    # class_weight=None
)

decision_tree.fit(X_train, y_train)

print("train score : {}".format(decision_tree.score(X_train, y_train)))
print("val score : {}".format(decision_tree.score(X_test, y_test)))

# 문장 분류하기
texts2 = []

f = open('../rhino/review_sent_rhino_1126.csv', 'r')
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

with open('sample_test.csv','w', newline='') as f:
    makewrite = csv.writer(f)
    for value in prediction:
        makewrite.writerow(value)

# 결과문장
count = 0
arr = []
fff = open('sample_test.csv', 'r')
for line in fff.readlines():
    oneline = line.replace("\n", "").split(",")
    count = count + 1
    if oneline[2] == '1':
        arr.append(count)
# oneline[0] == '1' or oneline[1] == '1' or

with open('decisionTree_result.csv', 'w', newline='') as ffff:
    makewrite = csv.writer(ffff)
    for value in arr:
        makewrite.writerow(texts2[value])
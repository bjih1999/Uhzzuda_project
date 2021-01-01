import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

texts = []
f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts.append(oneline)

texts = texts[:5000]
print(len(texts))

dataset = []
model = Doc2Vec.load('../imbedding/doc2vec_v300_w10')
for i in texts:
    dataset.append(model.infer_vector(i))

ff = open('new_label.csv', 'r')
arr = []
label = []
reader = csv.reader(ff)
for row in reader:
    arr.append(row)

for i in range(len(arr)):
    line = []
    for j in range(len(arr[i])):
        if arr[i][j] == '':
            arr[i][j] = 0
        if arr[i][j] == '1':
            arr[i][j] = 1
        if arr[i][j] == '0':
            arr[i][j] = 0
        line.append(arr[i][j])
    label.append(line)

labels = np.array(label)

df = pd.DataFrame(dataset)
X = df.iloc[:, 0:300]

lf = pd.DataFrame(labels)
y = lf.iloc[:, 0:7]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.125, random_state=3)

decision_tree = DecisionTreeClassifier(
    max_depth=10,
    # criterion='entropy',
    min_samples_split=20,
    # random_state=0,
    # min_samples_split=2,
    # min_samples_leaf=5,
    # max_features=None,
    # max_leaf_nodes=None,
    # class_weight=None
)

for i in range(0, 7):
    print(i, ' iteration')
    decision_tree.fit(X_train, y_train[:][i])

    print("train score : {}".format(decision_tree.score(X_train, y_train[:][i])))
    print("val score : {}".format(decision_tree.score(X_test, y_test[:][i])))

    validation_prediction = decision_tree.predict(X_test)
    f1 = recall_score(validation_prediction, y_test[:][i])
    print("f1 : ", f1)

    filePath = 'knnClassifier_' + str(i) + '_25000.csv'
    outfile = open(filePath, 'w', newline='')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: x, true_data))
    outfile.close()

'''
# 문장 분류하기
texts2 = []

f = open('../rhino/review_sent_rhino_1126.csv', 'r')
for line4 in f.readlines():
    oneline3 = line4.replace("\n", "").split(",")
    texts2.append(oneline3)

wtf = []
for i in texts2:
    wtf.append(model.infer_vector(i))

prediction = decision_tree.predict(wtf)
prediction = prediction.T
# print('prediction! \n', prediction)

with open('decisionTree_vec300&label2300.csv','w', newline='') as f:
    makewrite = csv.writer(f)
    for value in prediction:
        makewrite.writerow(value)

# 결과문장
count = 0
arr = []
fff = open('decisionTree_vec300&label2300.csv', 'r')
for line in fff.readlines():
    oneline = line.replace(",", "")
    if oneline == '1':
        arr.append(count)
    count = count + 1

with open('decisionTree_professor+.csv', 'w', newline='') as ffff:
    makewrite = csv.writer(ffff)
    for value in arr:
        makewrite.writerow(texts2[value])
'''

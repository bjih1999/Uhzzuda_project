import pandas._libs.sparse
import pandas.core.arrays.sparse
import scipy.sparse
from skmultilearn.adapt import MLTSVM
from sklearn.model_selection import GridSearchCV, train_test_split
from gensim.models.doc2vec import Doc2Vec
import pandas as pd
import numpy as np
import csv
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

preprocessed_texts5000 = []
f = open('rhino/preprocessed_review_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts5000.append(oneline)
    preprocessed_texts5000 = preprocessed_texts5000[:5000]
print('training_points : ', len(preprocessed_texts5000))

preprocessed_texts = []
f = open('rhino/preprocessed_review_rhino_1126.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)

embedded_texts = []
model = Doc2Vec.load('imbedding/doc2vec_v300_w10')
for i in preprocessed_texts5000:
    embedded_texts.append(model.infer_vector(i))

size = (len(preprocessed_texts5000), 13)
labels = np.zeros(size, dtype=int)
ff = open('classifier/label_5000.csv', 'r')
arr = []
label = []
reader = csv.reader(ff)
for row in reader:
    arr.append(row)

while len(arr) < 5000:
    arr.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

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
X = pd.DataFrame(embedded_texts)
y = pd.DataFrame(labels)

df_data = X.iloc[:, 0:300]
df_labels = y.iloc[:, 0:13]

X_trainTemp, X_testTemp, y_trainTemp, y_testTemp = \
    train_test_split(df_data, df_labels, test_size=0.1, random_state=52, shuffle=False)

# X_train = np.array(X_trainTemp)
# y_train = np.array(y_trainTemp)
# X_test = np.array(X_testTemp)

print("1")

X_sparse = scipy.sparse.csr_matrix(X_trainTemp)
y_sparse = scipy.sparse.csr_matrix(y_trainTemp)
xte = scipy.sparse.csr_matrix(X_testTemp)

print("2")

# print('X_trainTemp !\n', X_trainTemp)
# print('X_train !\n', xtr)

mlsvm = MLTSVM(c_k = 2**-1)
print("3")


mlsvm.fit(X_sparse, y_sparse)
print("4")


pred = mlsvm.predict(xte)
print("Accuracy = ", accuracy_score(y_testTemp, pred))
print("Precision = ", precision_score(y_testTemp, pred, average='macro'))
print("Recall = ", recall_score(y_testTemp, pred, average='micro'))

# 문장 분류하기
##원본 문장
review_texts = []

f = open('rhino/review_sent_rhino_1126.csv', 'r')
for line4 in f.readlines():
    oneline3 = line4.replace("\n", "").split(",")
    review_texts.append(oneline3)

wtf = []
for i in preprocessed_texts5000:
    wtf.append(model.infer_vector(i))

wtf = np.array(wtf)

prediction = mlsvm.predict(wtf)

with open('dd.csv','w', newline='') as f:
    makewrite = csv.writer(f)
    for value in prediction:
        makewrite.writerow(value)

# https://towardsdatascience.com/knn-algorithm-what-when-why-how-41405c16c36f
# https://www.geeksforgeeks.org/an-introduction-to-multilabel-classification/
# https://medium.com/x8-the-ai-community/knn-classification-algorithm-in-python-65e413e1cea0
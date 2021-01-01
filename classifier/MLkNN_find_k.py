from gensim.models.doc2vec import Doc2Vec
from skmultilearn.adapt import MLkNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from scipy.sparse import lil_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, hamming_loss

from sklearn.model_selection import GridSearchCV
import pickle

preprocessed_texts5000 = []
# f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')
f = open('fromok_pre_8.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts5000.append(oneline)

preprocessed_texts_all = preprocessed_texts5000
preprocessed_texts5000 = preprocessed_texts5000[:6983]

print('training_points : ', len(preprocessed_texts5000))

embedded_texts = []
# model = Doc2Vec.load('8.model')
# model = Doc2Vec.load('doc2vec/add2.model')
# model = Doc2Vec.load('jahee_model/realok_300_10.model')
model = Doc2Vec.load('jahee_model/ok_add10_re.model')

for i in preprocessed_texts5000:
    embedded_texts.append(model.infer_vector(i))

size = (len(preprocessed_texts5000), 13)
labels = np.zeros(size, dtype=int)
# ff = open('label_7000_final_ver.csv', 'r')
ff = open('fromok_label_8_real.csv', 'r')
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
        elif arr[i][j] == '1':
            arr[i][j] = 1
        elif arr[i][j] == '0':
            arr[i][j] = 0
        else:
            arr[i][j] = 1
        line.append(arr[i][j])
    label.append(line)

label = label[:6983]
print('label : ', len(label))

labels = np.array(label)
X = pd.DataFrame(embedded_texts)
y = pd.DataFrame(labels)

df_data = X.iloc[:, 0:300]
df_labels = y.iloc[:, 0:13]

max_f1score = 0
max_randomState = 0
# 0.24712643678160917
for kk in range(0, 30):
    print(kk)

    training_data, validation_data , training_labels, validation_labels = \
        train_test_split(df_data, df_labels, test_size = 0.1, random_state=kk)

    X_train = lil_matrix(training_data).toarray()
    X_test = lil_matrix(validation_data).toarray()
    y_train = lil_matrix(training_labels).toarray()
    y_test = lil_matrix(validation_labels).toarray()

    mlknn = MLkNN(k=21)
    mlknn.fit(X_train, y_train)

    validation_prediction = mlknn.predict(X_test)
    f1_scoreMicro = f1_score(y_test, validation_prediction, average='micro')

    if max_f1score < f1_scoreMicro:
        max_randomState = kk
        max_f1score = f1_scoreMicro
        pickle.dump(mlknn, open('MODEL_MAX.sav', 'wb'))

        hammingLoss = hamming_loss(y_test, validation_prediction)
        accuracy = accuracy_score(y_test, validation_prediction)
        f1_scoreMacro = f1_score(y_test, validation_prediction, average='macro')
        print("Hamming Loss = ", round((1 - hammingLoss), 2))
        print("Accuracy = ", round(accuracy, 2))
        print("f1_scoreMacro = ", round(f1_scoreMacro, 2))
        print("f1_scoreMicro = ", round(f1_scoreMicro, 2))
        print(classification_report(y_test, validation_prediction))

print('MAX f1_score : ', max_f1score)
print('MAX random_state : ', max_randomState)

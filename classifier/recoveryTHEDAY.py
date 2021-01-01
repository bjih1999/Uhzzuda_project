from gensim.models.doc2vec import Doc2Vec
from skmultilearn.adapt import MLkNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, hamming_loss
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle

preprocessed_texts5000 = []
f = open('3_tunning_shuffled_pre_review_rhino_1203_add2000.csv', 'r')
for line in f.readlines():
    # oneline = line.replace('숙,제', ',숙제,')
    # oneline = oneline.replace('기,말', ',기말,')
    # oneline = oneline.replace('기,출', ',기출,')
    # oneline = oneline.replace('기,대', ',기대,')
    # oneline = oneline.replace('기,초', ',기초,')
    # oneline = oneline.replace('알,채', ',안,채')
    # oneline = oneline.replace('안채,워', ',안,채우,')
    # oneline = oneline.replace('재수,강', ',재수강,')
    # oneline = oneline.replace('싸,강', ',싸강,')
    # oneline = oneline.replace('인,강', ',싸강,')
    # oneline = oneline.replace('연,강', ',연강,')
    # oneline = oneline.replace('강,의', ',강의,')
    # oneline = oneline.replace('빡,세', ',빡세,')
    # oneline = oneline.replace('빡,쎄', ',빡세,')
    # oneline = oneline.replace('빡,셉', ',빡세,')
    # oneline = oneline.replace('빡,셉', ',빡세,')
    # oneline = oneline.replace('빡,치', ',빡치,')
    # oneline = oneline.replace('빡,침', ',빡치,')
    # oneline = oneline.replace('외,우', ',외우,')
    # oneline = oneline.replace('외,울', ',외우,')
    # oneline = oneline.replace('외,운', ',외우,')
    # oneline = oneline.replace('전범,위', ',전체,범위,')
    # oneline = oneline.replace('저,범위', ',전체,범위,')
    # oneline = oneline.replace('오프,', ',오프라인,')
    # oneline = oneline.replace('오프라,', ',오프라인,')
    # oneline = oneline.replace('온오,프', ',온오프,')
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts5000.append(oneline)

preprocessed_texts_all = preprocessed_texts5000
preprocessed_texts5000 = preprocessed_texts5000[:7000]

print('training_points : ', len(preprocessed_texts5000))

embedded_texts = []
# model = Doc2Vec.load('8.model')
# model = Doc2Vec.load('doc2vec/add2.model')
model = Doc2Vec.load('doc2vec_1202.model')

for i in preprocessed_texts5000:
    embedded_texts.append(model.infer_vector(i))

size = (len(preprocessed_texts5000), 13)
labels = np.zeros(size, dtype=int)
# ff = open('label_7000_final_ver.csv', 'r')
ff = open('label_7000_final_ver.csv', 'r')
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

label = label[:7000]
print('label : ', len(label))

labels = np.array(label)
X = pd.DataFrame(embedded_texts)
y = pd.DataFrame(labels)

df_data = X.iloc[:, 0:300]
df_labels = y.iloc[:, 0:13]

training_data, validation_data , training_labels, validation_labels = \
    train_test_split(df_data, df_labels, test_size = 0.1, random_state=0)

X_train = lil_matrix(training_data).toarray()
y_train = lil_matrix(validation_data).toarray()
X_test = lil_matrix(training_labels).toarray()
y_test = lil_matrix(validation_labels).toarray()

mlknn = MLkNN(k=1)
mlknn.fit(X_train, X_test)

pickle.dump(mlknn, open('mlknn_THEDAY_compare2_model.sav', 'wb'))

training_data, validation_data, training_labels, validation_labels = \
    train_test_split(df_data, df_labels, test_size=0.1, shuffle=True)

X_train = lil_matrix(training_data).toarray()
y_train = lil_matrix(validation_data).toarray()
X_test = lil_matrix(training_labels).toarray()
y_test = lil_matrix(validation_labels).toarray()

validation_prediction = mlknn.predict(y_train)
f1_scoreMicro = f1_score(y_test, validation_prediction, average='micro')

hammingLoss = hamming_loss(y_test, validation_prediction)
accuracy = accuracy_score(y_test, validation_prediction)
f1_scoreMacro = f1_score(y_test, validation_prediction, average='macro')
f1_scoreMicro = f1_score(y_test, validation_prediction, average='micro')
print("Hamming Loss = ", round((1 - hammingLoss), 2))
print("Accuracy = ", round(accuracy, 2))
print("f1_scoreMacro = ", round(f1_scoreMacro, 2))
print("f1_scoreMicro = ", round(f1_scoreMicro, 2))
print(classification_report(y_test, validation_prediction))
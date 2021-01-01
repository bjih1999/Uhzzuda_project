from gensim.models.doc2vec import Doc2Vec
from skmultilearn.adapt import MLkNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping

preprocessed_texts5000 = []
f = open('3_tunning_shuffled_pre_review_rhino_1203_add2000.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts5000.append(oneline)

preprocessed_texts_all = preprocessed_texts5000
preprocessed_texts5000 = preprocessed_texts5000[:7000]

print('training_points : ', len(preprocessed_texts5000))

embedded_texts = []
model = Doc2Vec.load('doc2vec_new.model')

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

K.clear_session()

lstm_model = Sequential()
lstm_model.add(LSTM(10))
lstm_model.add(Dense(13, activation='softmax'))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

lstm_model.fit(X_train, X_test, epochs=100,
          batch_size=30, verbose=1)

lstm_model.summary()

scores = lstm_model.evaluate(y_train, y_test)
print(scores)

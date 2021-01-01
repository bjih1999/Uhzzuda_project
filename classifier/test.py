from preprocessing import preprocess
from gensim.models.doc2vec import Doc2Vec
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, hamming_loss
from scipy.sparse import lil_matrix
import pandas as pd
import csv
import numpy as np

preprocessed_texts = []
f = open('fromok_pre_8.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)

preprocessed_texts_all = preprocessed_texts
preprocessed_texts5000 = preprocessed_texts[:6983]

print('training_points : ', len(preprocessed_texts5000))

embedded_texts = []
# model = Doc2Vec.load('jahee_model/doc.model')
model = Doc2Vec.load('doc2vec_1202.model')

for i in preprocessed_texts5000:
    embedded_texts.append(model.infer_vector(i))

size = (len(preprocessed_texts5000), 13)
labels = np.zeros(size, dtype=int)
# ff = open('label_7000_fix_ver.csv', 'r')
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
        if arr[i][j] == '1':
            arr[i][j] = 1
        if arr[i][j] == '0':
            arr[i][j] = 0

        line.append(arr[i][j])
    label.append(line)

label = label[:6983]
print('label : ', len(label))

labels = np.array(label)
X = pd.DataFrame(embedded_texts)
y = pd.DataFrame(labels)

df_data = X.iloc[:, 0:300]
df_labels = y.iloc[:, 0:13]

training_data, validation_data , training_labels, validation_labels = \
    train_test_split(df_data, df_labels, test_size = 0.1)

X_train = lil_matrix(training_data).toarray()
y_train = lil_matrix(validation_data).toarray()
X_test = lil_matrix(training_labels).toarray()
y_test = lil_matrix(validation_labels).toarray()
mlknn = pickle.load(open('mlknn1_model.sav', 'rb'))

validation_prediction = mlknn.predict(y_train)

df = pd.DataFrame(validation_prediction.todense())
df.to_csv('mlknn_predictions_vec300.csv', index=False, header=None)

hammingLoss = hamming_loss(y_test, validation_prediction)
accuracy = accuracy_score(y_test, validation_prediction)
f1_scoreMacro = f1_score(y_test, validation_prediction, average='macro')
f1_scoreMicro = f1_score(y_test, validation_prediction, average='micro')
print("Hamming Loss = ", round((1 - hammingLoss), 2))
print("Accuracy = ", round(accuracy, 2))
print("f1_scoreMacro = ", round(f1_scoreMacro, 2))
print("f1_scoreMicro = ", round(f1_scoreMicro, 2))
print(classification_report(y_test, validation_prediction))
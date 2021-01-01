from gensim.models.doc2vec import Doc2Vec
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

preprocessed_texts5000 = []
# f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')
f = open('tunning_shuffled_pre_review_rhino_1203.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts5000.append(oneline)

preprocessed_texts5000 = preprocessed_texts5000[:5000]

f = open('temp_data_fix.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts5000.append(oneline)
preprocessed_texts5000 = preprocessed_texts5000[:7000]

print('training_points : ', len(preprocessed_texts5000))

embedded_texts = []
model = Doc2Vec.load('../imbedding/doc2vec_v300_w10')
for i in preprocessed_texts5000:
    embedded_texts.append(model.infer_vector(i))

size = (len(preprocessed_texts5000), 13)
labels = np.zeros(size, dtype=int)
ff = open('label_5000.csv', 'r')
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

while len(label) < 5000:
    label.append(['', '', '', '', '', '', '', '', '', '', '', '', ''])

ff = open('label_7708.csv', 'r')
arr = []
reader = csv.reader(ff)
for row in reader:
    label.append(row)

label = label[:7000]

while len(label) < 7000:
    label.append(['', '', '', '', '', '', '', '', '', '', '', '', ''])

print('label : ', len(label))

labels = np.array(label)
X = pd.DataFrame(embedded_texts)
y = pd.DataFrame(labels)

df_data = X.iloc[:, 0:300]
df_labels = y.iloc[:, 0:13]

knn_classifier = KNeighborsClassifier(
    n_neighbors = 100,
    # weights = 'distance',
)

training_data, validation_data , training_labels, validation_labels = \
    train_test_split(df_data, df_labels, test_size = 0.1)

review_texts = []

f = open('../rhino/review_sent_rhino_1126.csv', 'r')
for line4 in f.readlines():
    oneline3 = line4.replace("\n", "").split(",")
    review_texts.append(oneline3)

wtf = []
for i in preprocessed_texts5000:
    wtf.append(model.infer_vector(i))


knn_classifier.fit(training_data, training_labels)
validation_prediction = knn_classifier.predict(validation_data)

print("Accuracy = ", accuracy_score(validation_labels, validation_prediction))
print("Precision = ", precision_score(validation_labels, validation_prediction, average='macro', zero_division=1))
print("Recall = ", recall_score(validation_labels, validation_prediction, average=None))
print("f1 score = ", f1_score(validation_labels, validation_prediction))

validation_prediction = knn_classifier.predict(wtf)

for i in range(0, 13):
    true_data = []
    count = 0
    for j in range(len(validation_prediction)):
        if validation_prediction[j] == 1:
            true_data.append(review_texts[count])
        count = count + 1

    filePath = 'knnClassifier_' + str(i) + 'multilabel.csv'
    outfile = open(filePath, 'w', newline='')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: x, true_data))
    outfile.close()
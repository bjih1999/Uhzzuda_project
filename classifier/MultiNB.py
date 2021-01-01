from gensim.models.doc2vec import Doc2Vec
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

preprocessed_texts2300 = []
f = open('tunning_shuffled_pre_review_rhino_1203.csv', 'r')
# f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')

for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts2300.append(oneline)
preprocessed_texts2300 = preprocessed_texts2300[:5000]
print('training_points : ', len(preprocessed_texts2300))

preprocessed_texts = []
# f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')
f = open('tunning_shuffled_pre_review_rhino_1203.csv', 'r')

for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)

embedded_texts = []
model = Doc2Vec.load('../imbedding/doc2vec_v300_w10')
for i in preprocessed_texts2300:
    embedded_texts.append(model.infer_vector(i))

size = (len(preprocessed_texts2300), 13)
labels = np.zeros(size, dtype=int)
ff = open('label_5000.csv', 'r')
arr = []
label = []
reader = csv.reader(ff)
for row in reader:
    arr.append(row)

while len(arr) < 5000:
    arr.append(['', '', '', '', '', '', '', '', '', '', '', '', ''])

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

NB_classifier = MultinomialNB(class_prior=[0.01, 0.5])

for i in range(len(X)):
    if X[i] < 0:
        X[i] = abs(X[i])

training_data, validation_data , training_labels, validation_labels = \
    train_test_split(df_data, df_labels, test_size = 0.1, random_state = 0)

review_texts = []

f = open('../rhino/review_sent_rhino_1126.csv', 'r')
for line4 in f.readlines():
    oneline3 = line4.replace("\n", "").split(",")
    review_texts.append(oneline3)

wtf = []
for i in preprocessed_texts:
    wtf.append(model.infer_vector(i))

true_data = []
count = 0

for i in range(0, 13):
    print(i)

    NB_classifier.fit(training_data, training_labels[:][i])
    validation_prediction = NB_classifier.predict(validation_data)

    print("Accuracy = ", accuracy_score(validation_labels[:][i], validation_prediction))
    print("Precision = ",
          precision_score(validation_labels[:][i], validation_prediction, average='macro', zero_division=1))
    print("Recall = ", recall_score(validation_labels[:][i], validation_prediction, ))
    print("f1 score = ", f1_score(validation_labels[:][i], validation_prediction))

    validation_prediction = NB_classifier.predict(wtf)

    for j in range(len(validation_prediction)):
        if validation_prediction[j] == 1:
            true_data.append(review_texts[count])
        count = count + 1

    filePath = 'NB_' + str(i) + '_wtf_tunning.csv'
    outfile = open(filePath, 'w', newline='')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: x, true_data))
    outfile.close()

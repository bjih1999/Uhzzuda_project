from gensim.models.doc2vec import Doc2Vec
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

preprocessed_texts2300 = []
f = open('classifier/2_tunning_shuffled_pre_review_rhino_1203.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts2300.append(oneline)
preprocessed_texts2300 = preprocessed_texts2300[:5000]
print('training_points : ', len(preprocessed_texts2300))

preprocessed_texts = []
f = open('classifier/2_tunning_shuffled_pre_review_rhino_1203.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)

embedded_texts = []
model = Doc2Vec.load('jaebal/doc2vec/4.model')
for i in preprocessed_texts2300:
    embedded_texts.append(model.infer_vector(i))

size = (len(preprocessed_texts2300), 13)
labels = np.zeros(size, dtype=int)
ff = open('classifier/label_7708_last.csv', 'r')
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

NB_classifier = GaussianNB()

training_data, validation_data , training_labels, validation_labels = \
    train_test_split(df_data, df_labels, test_size = 0.1, random_state = 52)

NB_classifier.fit(training_data, training_labels)

pred = NB_classifier.predict(validation_data[:][8])
print("Accuracy = ", accuracy_score(validation_labels[:][8], pred))
print("Precision = ", precision_score(validation_labels[:][8], pred, average='macro', zero_division=1))
print("Recall = ", recall_score(validation_labels[:][8], pred, average='micro'))
print("f1 score = ", f1_score(validation_labels[:][8], pred))

# 문장 분류하기
##원본 문장
review_texts = []

f = open('rhino/review_sent_rhino_1126.csv', 'r')
for line4 in f.readlines():
    oneline3 = line4.replace("\n", "").split(",")
    review_texts.append(oneline3)

wtf = []
for i in preprocessed_texts:
    wtf.append(model.infer_vector(i))

prediction = NB_classifier.predict(wtf)
# 너무 느리면 prediction = knn_classifier.predict(validation_data)

# 6항목을 정답으로 분류한 결과 문장 (6은 교수님+)
count = 0
with open('classifier/NBClassifier_class+_new.csv', 'w', newline='') as ffff:
    makewrite = csv.writer(ffff)
    for row_prediction in prediction:
        if row_prediction == 1:
            makewrite.writerow(review_texts[count])
        count = count + 1

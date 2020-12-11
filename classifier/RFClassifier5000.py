from gensim.models.doc2vec import Doc2Vec
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

preprocessed_texts2300 = []
f = open('classifier/3_tunning_shuffled_pre_review_rhino_1203_add2000.csv', 'r')
for line in f.readlines():
    oneline = line.replace('숙,제', ',숙제,')
    oneline = oneline.replace('기,말', ',기말,')
    oneline = oneline.replace('기,출', ',기출,')
    oneline = oneline.replace('기,대', ',기대,')
    oneline = oneline.replace('기,초', ',기초,')
    oneline = oneline.replace('알,채', ',안,채')
    oneline = oneline.replace('안채,워', ',안,채우')
    oneline = oneline.replace('재수,강', ',재수강,')
    oneline = oneline.replace('싸,강', ',싸강,')
    oneline = oneline.replace('인,강', ',싸강,')
    oneline = oneline.replace('강,의', ',강의,')
    oneline = oneline.replace('빡,세', ',빡세,')
    oneline = oneline.replace('빡,쎄', ',빡세,')
    oneline = oneline.replace('빡,셉', ',빡세,')
    oneline = oneline.replace('빡,셉', ',빡세,')
    oneline = oneline.replace('빡,치', ',빡치,')
    oneline = oneline.replace('빡,침', ',빡치,')
    oneline = oneline.replace('외,우', ',외우,')
    oneline = oneline.replace('외,울', ',외우,')
    oneline = oneline.replace('외,운', ',외우,')
    oneline = oneline.replace('전범,위', ',전체,범위,')
    oneline = oneline.replace('저,범위', ',전체,범위,')
    oneline = oneline.replace('오프,', ',오프라인,')
    oneline = oneline.replace('오프라,', ',오프라인,')
    oneline = oneline.replace('온오,프', ',온오프,')
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts2300.append(oneline)
preprocessed_texts2300 = preprocessed_texts2300[:7000]
print('training_points : ', len(preprocessed_texts2300))

preprocessed_texts = []
f = open('classifier/3_tunning_shuffled_pre_review_rhino_1203_add2000.csv', 'r')
for line in f.readlines():
    oneline = line.replace('숙,제', ',숙제,')
    oneline = oneline.replace('기,말', ',기말,')
    oneline = oneline.replace('기,출', ',기출,')
    oneline = oneline.replace('기,대', ',기대,')
    oneline = oneline.replace('기,초', ',기초,')
    oneline = oneline.replace('알,채', ',안,채')
    oneline = oneline.replace('안채,워', ',안,채우')
    oneline = oneline.replace('재수,강', ',재수강,')
    oneline = oneline.replace('싸,강', ',싸강,')
    oneline = oneline.replace('인,강', ',싸강,')
    oneline = oneline.replace('강,의', ',강의,')
    oneline = oneline.replace('빡,세', ',빡세,')
    oneline = oneline.replace('빡,쎄', ',빡세,')
    oneline = oneline.replace('빡,셉', ',빡세,')
    oneline = oneline.replace('빡,셉', ',빡세,')
    oneline = oneline.replace('빡,치', ',빡치,')
    oneline = oneline.replace('빡,침', ',빡치,')
    oneline = oneline.replace('외,우', ',외우,')
    oneline = oneline.replace('외,울', ',외우,')
    oneline = oneline.replace('외,운', ',외우,')
    oneline = oneline.replace('전범,위', ',전체,범위,')
    oneline = oneline.replace('저,범위', ',전체,범위,')
    oneline = oneline.replace('오프,', ',오프라인,')
    oneline = oneline.replace('오프라,', ',오프라인,')
    oneline = oneline.replace('온오,프', ',온오프,')
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts.append(oneline)

embedded_texts = []
model = Doc2Vec.load('jaebal/doc2vec/8.model')
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

# while len(arr) < 5000:
#     arr.append(['', '', '', '', '', '', '', '', '', '', '', '', ''])
arr = arr[:7000]

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

RF_classifier = RandomForestClassifier()

training_data, validation_data , training_labels, validation_labels = \
    train_test_split(df_data, df_labels, test_size = 0.1, random_state = 52)

RF_classifier.fit(training_data, training_labels[:][12])

pred = RF_classifier.predict(validation_data)
print("Accuracy = ", accuracy_score(validation_labels[:][12], pred))
print("Precision = ", precision_score(validation_labels[:][12], pred, average='macro', zero_division=1))
print("Recall = ", recall_score(validation_labels[:][12], pred, average='micro'))
print("f1 score = ", f1_score(validation_labels[:][12], pred))

# 문장 분류하기
##원본 문장
review_texts = []

f = open('classifier/shuffled_review_sent_rhino_1126_add2000.csv', 'r')
for line4 in f.readlines():
    oneline3 = line4.replace("\n", "").split(",")
    review_texts.append(oneline3)

wtf = []
for i in preprocessed_texts:
    wtf.append(model.infer_vector(i))

prediction = RF_classifier.predict(wtf)
# 너무 느리면 prediction = knn_classifier.predict(validation_data)

# 6항목을 정답으로 분류한 결과 문장 (6은 교수님+)
count = 0
with open('classifier/RFClassifier_team+_tuned.csv', 'w', newline='') as ffff:
    makewrite = csv.writer(ffff)
    for row_prediction in prediction:
        if row_prediction == 1:
            makewrite.writerow(review_texts[count])
        count = count + 1

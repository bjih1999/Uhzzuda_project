from gensim.models.doc2vec import Doc2Vec
from skmultilearn.adapt import MLkNN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle

embedding_model = Doc2Vec.load('doc2vec_new.model')
classifier_model = pickle.load(open('mlknn1_model.sav', 'rb'))
# 
temp_sentence = []
preprocessed_texts5000 = []
# f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')
f = open('3_tunning_shuffled_pre_review_rhino_1203_add2000.csv', 'r')
for line in f.readlines():
    oneline = line.replace('\n', '').split(',')
    oneline = list(filter(None, oneline))
    preprocessed_texts5000.append(oneline)

preprocessed_sentence = preprocessed_texts5000[:20]

review_texts = []

f = open('shuffled_review_sent_rhino_1126_add2000.csv', 'r')
for line4 in f.readlines():
    oneline3 = line4.replace("\n", "").split(",")
    oneline3 = list(filter(None, oneline3))

    review_texts.append(oneline3)

review_texts = review_texts[:20]

for i in preprocessed_sentence:
    temp_sentence.append(embedding_model.infer_vector(i))

temp_sentence = lil_matrix(temp_sentence).toarray()
predictions = classifier_model.predict(temp_sentence)

df = pd.DataFrame(predictions.todense())
df.to_csv('temp_prediction.csv', index=False, header=None)

arr = []
label = []
ff = open('temp_prediction.csv', 'r')
reader = csv.reader(ff)
for row in reader:
    arr.append(row)

for i in range(len(df)):
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

for i in range(len(df)):
    print(review_texts[i], ' : ', end='')
    cnt = 0
    if label[i][0] == 1:
        print('과제+', end=' ')
        cnt = cnt+1
    if label[i][1] == 1:
        print('과제-', end=' ')
        cnt = cnt+1
    if label[i][2] == 1:
        print('성적+', end=' ')
        cnt = cnt+1
    if label[i][3] == 1:
        print('성적-', end=' ')
        cnt = cnt+1
    if label[i][4] == 1:
        print('시험+', end=' ')
        cnt = cnt+1
    if label[i][5] == 1:
        print('시험-', end=' ')
        cnt = cnt+1
    if label[i][6] == 1:
        print('교수+', end=' ')
        cnt = cnt+1
    if label[i][7] == 1:
        print('교수-', end=' ')
        cnt = cnt+1
    if label[i][8] == 1:
        print('강의+', end=' ')
        cnt = cnt+1
    if label[i][9] == 1:
        print('강의-', end=' ')
        cnt = cnt+1
    if label[i][10] == 1:
        print('부담+', end=' ')
        cnt = cnt+1
    if label[i][11] == 1:
        print('부담-', end=' ')
        cnt = cnt+1
    if label[i][12] == 1:
        print('팀플+', end=' ')
        cnt = cnt+1
    if cnt == 0:
        print('X', end='')
    print()

from preprocessing import preprocess
from gensim.models.doc2vec import Doc2Vec
import pickle
from scipy.sparse import lil_matrix
import pandas as pd
import csv

review_texts = []
# f = open('진로와직업선택_박윤희.csv', 'r')
# f = open('shuffled_review_sent_rhino_1126_add2000.csv', 'r')
# f = open('../rhino/review_sent_rhino_1126.csv', 'r')
f = open('testtest.csv', 'r')
for line4 in f.readlines():
    line4 = line4.replace(',', '')
    oneline3 = line4.replace("\n", "").split(",")
    review_texts.append(oneline3)

# nature_sentence = review_texts[1:4]
nature_sentence = review_texts
preprocessed_sentence = []
lecture_len = []
for i in nature_sentence:
    temp, temp_len = preprocess(i[0])
    preprocessed_sentence.append(temp)
    lecture_len.append(temp_len)

model = Doc2Vec.load('jahee_model/5_add10.model')

embedded_sentence = []
for lecture_sentence in preprocessed_sentence:
    for i in range(len(lecture_sentence)):
        embedded_sentence.append(model.infer_vector(lecture_sentence[i]))

classifier_model = pickle.load(open('LSTM_model_add10.sav', 'rb'))

predictions = classifier_model.predict(embedded_sentence)
label = predictions

i = 0
for index in range(len(nature_sentence)):
    for j in range(lecture_len[index]):
        cnt = 0
        print(preprocessed_sentence[index][j], end=" : ")
        if label[i][0] == 1:
            print('과제+', end=' ')
            cnt = cnt + 1
        if label[i][1] == 1:
            print('과제-', end=' ')
            cnt = cnt + 1
        if label[i][2] == 1:
            print('성적+', end=' ')
            cnt = cnt + 1
        if label[i][3] == 1:
            print('성적-', end=' ')
            cnt = cnt + 1
        if label[i][4] == 1:
            print('시험+', end=' ')
            cnt = cnt + 1
        if label[i][5] == 1:
            print('시험-', end=' ')
            cnt = cnt + 1
        if label[i][6] == 1:
            print('교수+', end=' ')
            cnt = cnt + 1
        if label[i][7] == 1:
            print('교수-', end=' ')
            cnt = cnt + 1
        if label[i][8] == 1:
            print('강의+', end=' ')
            cnt = cnt + 1
        if label[i][9] == 1:
            print('강의-', end=' ')
            cnt = cnt + 1
        if label[i][10] == 1:
            print('부담+', end=' ')
            cnt = cnt + 1
        if label[i][11] == 1:
            print('부담-', end=' ')
            cnt = cnt + 1
        if label[i][12] == 1:
            print('팀플+', end=' ')
            cnt = cnt + 1
        if cnt == 0:
            print('X', end='')
        print()
        i = i + 1

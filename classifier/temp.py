from preprocessing import preprocess
from gensim.models.doc2vec import Doc2Vec
import pickle
from scipy.sparse import lil_matrix
import pandas as pd
import csv

review_texts = []
# f = open('lecture_csv/진로와직업선택_박윤희.csv', 'r')
# f = open('lecture_csv/창의적사고와글쓰기_신경수.csv', 'r')
# f = open('lecture_csv/토론과커뮤니케이션_박삼열.csv', 'r')
f = open('lecture_csv/프랑스어권사회와문화_한선혜.csv', 'r')
# f = open('lecture_csv/합창과공동체인성_장세완.csv', 'r')
# f = open('lecture_csv/한반도의평화와선교.csv', 'r')
for line4 in f.readlines():
    line4 = line4.replace(',', '')
    oneline3 = line4.replace("\n", "").split(",")
    review_texts.append(oneline3)

# nature_sentence = review_texts
nature_sentence = review_texts
preprocessed_sentence = []
lecture_len = []
for i in nature_sentence:
    temp, temp_len = preprocess(i[0])
    preprocessed_sentence.append(temp)
    lecture_len.append(temp_len)

model = Doc2Vec.load('jahee_model/ok_add10_re.model')

embedded_sentence = []
for lecture_sentence in preprocessed_sentence:
    for i in range(len(lecture_sentence)):
        embedded_sentence.append(model.infer_vector(lecture_sentence[i]))

classifier_model = pickle.load(open('MODEL_MAX.sav', 'rb'))

temp_sentence = lil_matrix(embedded_sentence).toarray()
predictions = classifier_model.predict(temp_sentence)

df = pd.DataFrame(predictions.todense())
df.to_csv('temp_prediction2.csv', index=False, header=None)

arr = []
label = []
ff = open('temp_prediction2.csv', 'r')
reader = csv.reader(ff)
for row in reader:
    arr.append(row)

for i in range(len(df)):
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

i = 0
lecture_label = [0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0]
for index in range(len(nature_sentence)):
    print(nature_sentence[index], end=' : ')
    answer_label = [0, 0, 0, 0, 0,
                    0, 0 ,0 ,0, 0,
                    0, 0, 0]
    for j in range(lecture_len[index]):
        cnt = 0
        # print(preprocessed_sentence[index][j], end=" : ")
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
            print('X', end=' ')
        # print()
        for k in range(13):
            answer_label[k] = answer_label[k] or label[i][k]
        i = i + 1
    print()
    for num in range(13):
        lecture_label[num] = lecture_label[num] + answer_label[num]
    print(answer_label)
print()
print(lecture_label)
from preprocessing import preprocess
from gensim.models.doc2vec import Doc2Vec
import pickle
from scipy.sparse import lil_matrix
import pandas as pd
import csv

nature_sentence = input('입력 문장 : ')
# nature_sentence = '얻어가는 것은 어느 수업보다 많은 듯\n 중간고사 면접 기말고사 자유발표 교수님이 의도하는 바가 있으니\n그걸 정확히 캐치 하는것이 중요 (너가 하고싶은말 하는 시험아님)\n발표 경우 자유주제이지만 적절한 주제를 찾는게 가장 중요'
preprocessed_sentence = preprocess(nature_sentence)
# 교수님도 최고시구 수업내용도 좋습니다. 성적 잘받으려면 .

model = Doc2Vec.load('doc2vec_new.model')

embedded_sentence = []
for i in preprocessed_sentence:
    embedded_sentence.append(model.infer_vector(i))

classifier_model = pickle.load(open('mlknn4_model.sav', 'rb'))

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

for i in range(len(preprocessed_sentence)):
    cnt = 0
    print(preprocessed_sentence[i], ' : ', end='')
    if label[i][0] == 1:
        print('과제', end=' ')
        cnt = cnt+1
    if label[i][1] == 1:
        print('성적', end=' ')
        cnt = cnt+1
    if label[i][2] == 1:
        print('시험', end=' ')
        cnt = cnt+1
    if label[i][3] == 1:
        print('교수', end=' ')
        cnt = cnt+1
    if label[i][4] == 1:
        print('강의', end=' ')
        cnt = cnt+1
    if label[i][5] == 1:
        print('부담', end=' ')
        cnt = cnt+1
    if cnt == 0:
        print('X', end='')
    print()

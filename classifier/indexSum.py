import csv
import numpy as np

# preprocessed_texts5000 = []
# f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')
# for line in f.readlines():
#     oneline = line.replace('\n', '').split(',')
#     oneline = list(filter(None, oneline))
#     preprocessed_texts5000.append(oneline)
#
# preprocessed_texts5000 = preprocessed_texts5000[:5000]

ff = open('label_5000.csv', 'r')
arr = []
labels_text5000 = []
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
    labels_text5000.append(line)

# new_labels = np.array((5000, 7))
# print(labels_text5000)
real_labels = []
for i in range(len(labels_text5000)):
    # 과제 + 시험
    # 교수 + 강의
    # 부담 없애기
    new_labels = []

    l1 = labels_text5000[i][0] | labels_text5000[i][2] | labels_text5000[i][4]
    new_labels.append(l1)
    l2 = labels_text5000[i][1] | labels_text5000[i][3] | labels_text5000[i][5]
    new_labels.append(l2)
    # l3 = labels_text5000[i][2]
    # new_labels.append(l3)
    # l4 = labels_text5000[i][3]
    # new_labels.append(l4)
    l5 = labels_text5000[i][6] | labels_text5000[i][8]| labels_text5000[i][10]
    new_labels.append(l5)
    l6 = labels_text5000[i][7] | labels_text5000[i][9]| labels_text5000[i][11]
    new_labels.append(l6)
    # l7 = labels_text5000[i][12]
    # new_labels.append(l7)

    real_labels.append(new_labels)
with open('new2_label.csv', 'w', newline='') as f:
    makewrite = csv.writer(f)
    for value in real_labels:
        makewrite.writerow(value)

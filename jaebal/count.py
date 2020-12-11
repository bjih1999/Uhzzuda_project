import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import csv

num = 7000
texts = []
labels = np.zeros((7000,13), dtype=int)
print(len(labels))
f = open('label_7708.csv', 'r', encoding='UTF-8')
cnt = 0
counting = [0,0,0,0,0,0,0,0,0,0,0,0,0]
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    for i in range(13):
        if oneline[i] == '1':
            counting[i] += 1
            labels[cnt][i] = 1
    cnt = cnt + 1
    if cnt == num:
        break

print(labels)
print(counting)

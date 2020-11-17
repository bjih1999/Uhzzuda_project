import pandas as pd
import numpy as np

texts = []
f = open('../preprocess/preprocessed_review.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts.append(oneline)

f = open('../preprocess/preprocessed_review2.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    texts.append(oneline)


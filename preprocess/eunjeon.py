import pandas as pd
import numpy as np
import csv

texts = []
f = open('../reviewlist/reviewlist.csv', 'r')
for line in f.readlines():
    print(line)



from eunjeon import Mecab  # KoNLPy style mecab wrapper
tagger = Mecab()
for line in texts:
    print(tagger.pos(line))
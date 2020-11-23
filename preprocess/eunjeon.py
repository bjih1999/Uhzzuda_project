import pandas as pd
import numpy as np
import csv

reviewlist = []
with open('../reviewlist/reviewlist.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        reviewlist.append(row)


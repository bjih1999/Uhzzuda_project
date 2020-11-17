from konlpy.tag import Hannanum
from pykospacing import spacing
import csv
import re

hannanum = Hannanum()

reviewlist = []
with open('../reviewlist/reviewlist.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        reviewlist.append(row)


for review in reviewlist:
        print(review[0])


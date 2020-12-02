import csv
from random import seed, shuffle

reviewlist = []
preprolist = []

reviewlist1 = []
reviewlist2 = []
preprolist1 = []
preprolist2 = []

seed(0)

with open('shuffle\preprocessed_review_rhino_1126.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        preprolist.append(row)

with open('shuffle\\review_sent_rhino_1126.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        reviewlist.append(row)

reviewlist1 = reviewlist[:4000]
reviewlist2 = reviewlist[4000:]

preprolist1 = preprolist[:4000]
preprolist2 = preprolist[4000:]

numbers = list(range(len(reviewlist2)))

shuffle(numbers)

list1 = []
list2 = []
for num in numbers:
    list1.append(reviewlist2[num])
    list2.append(preprolist2[num])

dest_reviewlist = []
for row in reviewlist1:
    dest_reviewlist.append(row)
for row in list1:
    dest_reviewlist.append(row)

dest_preprolist = []
for row in preprolist1:
    dest_preprolist.append(row)
for row in list2:
    dest_preprolist.append(row)

with open('shuffle/shuffled_review_sent_rhino_1126.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    for row in dest_reviewlist:
        writer.writerow(row)

with open('shuffle/shuffled_preprocessed_review_rhino_1126.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    for row in dest_preprolist:
        writer.writerow(row)

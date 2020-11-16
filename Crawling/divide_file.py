import csv

f = open('lecturelist.csv', 'r')
lectures = csv.reader(f)
lecturelist = []

for lecture in lectures:
    lecturelist.append(lecture)

print(lecturelist[0][1:])
lecturelist = lecturelist[0][1:]
print(len(lecturelist))
print(len(lecturelist)//8)
first = lecturelist[0:len(lecturelist)//8]
second = lecturelist[len(lecturelist)//8+1:len(lecturelist)//8*2]
third = lecturelist[len(lecturelist)//8*2+1:len(lecturelist)//8*3]
fourth = lecturelist[len(lecturelist)//8*3+1:len(lecturelist)//8*4]
fifth = lecturelist[len(lecturelist)//8*4+1:len(lecturelist)//8*5]
sixth = lecturelist[len(lecturelist)//8*5+1:len(lecturelist)//8*6]
seventh = lecturelist[len(lecturelist)//8*6+1:len(lecturelist)//8*7]
eighth = lecturelist[len(lecturelist)//8*7+1:-1]

f.close()

with open('1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(first)

with open('2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(second)

with open('3.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(third)

with open('4.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fourth)

with open('5.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fifth)

with open('6.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(sixth)

with open('7.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(seventh)

with open('8.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(eighth)

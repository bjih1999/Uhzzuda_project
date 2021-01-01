import pandas as pd
import sklearn.model_selection
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

preprocessed_texts5000 = []
# f = open('../rhino/preprocessed_review_rhino_1126.csv', 'r')
f = open('tunning_shuffled_pre_review_rhino_1203.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    preprocessed_texts5000.append(oneline)

texts = preprocessed_texts5000[:5000]
print(len(texts))

dataset = []
model = Doc2Vec.load('../imbedding/doc2vec_v300_w10')
for i in texts:
    dataset.append(model.infer_vector(i))

ff = open('label_5000.csv', 'r')
arr = []
label = []
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
    label.append(line)

# label = label[:5000]
labels = np.array(label)

df = pd.DataFrame(dataset)
X = df.iloc[:, 0:300]

lf = pd.DataFrame(labels)
y = lf.iloc[:, 0:13]

training_data, validation_data, training_labels, validation_labels = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

random_tree = RandomForestClassifier(
    max_depth=2,
    # criterion='entropy',
    # min_samples_split=20,
    # random_state=0,
    # min_samples_split=2,
    # min_samples_leaf=5,
    # max_features=None,
    # max_leaf_nodes=None,
    # class_weight=None
)

review_texts = []

f = open('../rhino/review_sent_rhino_1126.csv', 'r')
for line4 in f.readlines():
    oneline3 = line4.replace("\n", "").split(",")
    review_texts.append(oneline3)

wtf = []
for i in preprocessed_texts5000:
    wtf.append(model.infer_vector(i))

for i in range(0, 13):
    print(i)
    random_tree.fit(training_data, training_labels[:][i])
    validation_prediction = random_tree.predict(validation_data)

    print("Accuracy = ", accuracy_score(validation_labels[:][i], validation_prediction))
    print("Precision = ", precision_score(validation_labels[:][i], validation_prediction, average='macro', zero_division=1))
    print("Recall = ", recall_score(validation_labels[:][i], validation_prediction))
    print("f1 score = ", f1_score(validation_labels[:][i], validation_prediction))

    print(validation_prediction)

    validation_prediction = random_tree.predict(wtf)
    true_data = []
    count = 0
    for j in range(len(validation_prediction)):
        if validation_prediction[j] == 1:
            true_data.append(review_texts[count])
        count = count + 1

    filePath = 'randomForest_' + str(i) + '_new2Label.csv'
    outfile = open(filePath, 'w', newline='')
    out = csv.writer(outfile)
    out.writerows(map(lambda x: x, true_data))
    outfile.close()
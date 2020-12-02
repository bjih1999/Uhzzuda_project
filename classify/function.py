def get10fold(data, turn):
    tot_length = len(data)
    each = int(tot_length / 10)  # 데이터를 10개 turn 으로 쪼개준다.
    mask = np.array([True if each * turn <= i < each * (turn + 1) else False
                     for i in list(range(tot_length))])  # 각 Turn 내에 있으면 True
    return data[~mask], data[mask]

def runCV(clf, data, labels):
    accuracies = []
    for i in range(10):
        data_tr, data_te = get10fold(data, i)
        labels_tr, labels_te = get10fold(labels, i)

        clf = clf.fit(data_tr, labels_tr)   # 모델 학습
        pred = clf.predict(data_te)         # 학습한 데이터로 모델 예측
        correct = pred == labels_te

        acc = sum(correct) / len(correct)
        accuracies.append(acc)

    return accuracies
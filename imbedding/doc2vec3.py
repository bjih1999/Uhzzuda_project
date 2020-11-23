import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

texts = []
f = open('../preprocess/preprocessed_review_jeol.csv', 'r')
for line in f.readlines():
    oneline = line.replace("\n", "").split(",")
    oneline = list(filter(None, oneline))
    if(len(oneline) > 1):
        texts.append(oneline)

# print(len(texts))

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

#1 doc2vec_model = Doc2Vec(documents, vector_size=50, window=7, min_count=30, workers=4)
doc2vec_model = Doc2Vec(documents, vector_size=50, window=7, min_count=40, workers=4)
doc2vec_model.save('doc2vec_jeol2.model')

doc2vec_model = Doc2Vec(documents, vector_size=50, window=5, min_count=30, workers=4)
doc2vec_model.save('doc2vec_jeol3.model')

doc2vec_model = Doc2Vec(documents, vector_size=50, window=10, min_count=30, workers=4)
doc2vec_model.save('doc2vec_jeol4.model')

doc2vec_model = Doc2Vec(documents, vector_size=70, window=10, min_count=30, workers=4)
doc2vec_model.save('doc2vec_jeol5.model')
"""
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]

doc2vec_model = Doc2Vec(documents, vector_size=50, window=7, min_count=30, workers=4)
doc2vec_model.save('doc2vec_sent2_new.model')

doc2vec_model = Doc2Vec(documents, vector_size=50, window=7, min_count=40, workers=4)
doc2vec_model.save('doc2vec_sent5_new.model')

doc2vec_model = Doc2Vec(documents, vector_size=50, window=7, min_count=20, workers=4)
doc2vec_model.save('doc2vec_sent1.model')

doc2vec_model = Doc2Vec(documents, vector_size=50, window=7, min_count=30, workers=4)
doc2vec_model.save('doc2vec_sent2.model')

doc2vec_model = Doc2Vec(documents, vector_size=70, window=7, min_count=20, workers=4)
doc2vec_model.save('doc2vec_sent3.model')

doc2vec_model = Doc2Vec(documents, vector_size=70, window=10, min_count=20, workers=4)
doc2vec_model.save('doc2vec_sent4.model')
"""


"""
data = pd.read_csv('../preprocess/entered_review.csv',engine='python',encoding='CP949')
data = data.transpose()
texts = data.index.tolist()

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
doc2vec_model = Doc2Vec(documents, vector_size=50, window=7, min_count=20, workers=4)
doc2vec_model.save('doc2vec_entered.model')
"""






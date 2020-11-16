from gensim.models.doc2vec import TaggedDocument

class Doc2VecCorpus:
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        with open(self.fname, encoding='utf-8') as file:
            for doc in file:

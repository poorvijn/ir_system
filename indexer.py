from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess

class Indexer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=preprocess)
        self.doc_vectors = None
        self.doc_ids = []

    def build_index(self, documents):
        """
        documents: list of (doc_id, text)
        """
        self.doc_ids = [doc_id for doc_id, _ in documents]
        texts = [text for _, text in documents]

        self.doc_vectors = self.vectorizer.fit_transform(texts)

    def get_vectors(self):
        return self.doc_vectors

    def get_vectorizer(self):
        return self.vectorizer

    def get_doc_ids(self):
        return self.doc_ids
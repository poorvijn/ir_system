from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    def __init__(self, vectorizer, doc_vectors, doc_ids):
        self.vectorizer = vectorizer
        self.doc_vectors = doc_vectors
        self.doc_ids = doc_ids

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])

        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = [(self.doc_ids[i], similarities[i]) for i in ranked_indices]

        return results
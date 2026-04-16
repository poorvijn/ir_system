import ir_datasets
from indexer import Indexer
from retreiver import Retriever
from evaluator import precision_at_k
import os
os.environ["IR_DATASETS_HOME"] = "./data"

def load_nfcorpus(max_docs=5000):
    dataset = ir_datasets.load("cranfield")

    docs = []
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= max_docs:
            break
        docs.append((doc.doc_id, doc.text))

    queries = {}
    for query in dataset.queries_iter():
        queries[query.query_id] = query.text

    qrels = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)

    return docs, queries, qrels


def main():
    print("Loading dataset...")
    docs, queries, qrels = load_nfcorpus(max_docs=5000)

    docs = docs[:5000]

    print("Building index...")
    indexer = Indexer()
    indexer.build_index(docs)

    retriever = Retriever(
        indexer.get_vectorizer(),
        indexer.get_vectors(),
        indexer.get_doc_ids()
    )

    print("\nRunning queries...\n")

    total_precision = 0
    count = 0

    for qid, query_text in list(queries.items())[:10]:  # test on 10 queries
        results = retriever.search(query_text, top_k=5)
        result_doc_ids = [doc_id for doc_id, _ in results]

        relevant_docs = qrels.get(qid, set())

        precision = precision_at_k(result_doc_ids, relevant_docs, k=5)

        print(f"Query: {query_text}")
        print(f"Top results: {result_doc_ids}")
        print(f"Precision@5: {precision:.2f}\n")

        total_precision += precision
        count += 1

    print(f"\nAverage Precision@5: {total_precision / count:.2f}")


if __name__ == "__main__":
    main()
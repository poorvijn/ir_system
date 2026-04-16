def precision_at_k(results, relevant_docs, k=5):
    """
    results: list of doc_ids returned
    relevant_docs: set of relevant doc_ids
    """
    results_k = results[:k]

    relevant_found = sum(1 for doc_id in results_k if doc_id in relevant_docs)

    return relevant_found / k


def reciprocal_rank_fusion(results, k=60):

    scores = {}
    doc_map = {}

    for docs in results:
        for rank, doc in enumerate(docs):

            key = doc.page_content

            if key not in scores:
                scores[key] = 0
                doc_map[key] = doc

            scores[key] += 1 / (k + rank + 1)

    reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_map[key] for key, _ in reranked]    
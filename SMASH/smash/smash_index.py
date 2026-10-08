from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy import sparse


@dataclass
class RetrievalHit:
    doc_id: str
    score: float
    index: int


class SparseDocumentIndex:


    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str],
                 pooled_vectors: sparse.spmatrix,
                 token_vectors: Sequence[Sequence[sparse.spmatrix]]):
        if len(doc_ids) != len(texts) or len(doc_ids) != pooled_vectors.shape[0]:
            raise ValueError("doc_ids, texts and pooled_vectors must have the same length")
        self.doc_ids = list(doc_ids)
        self.texts = list(texts)
        self.pooled = pooled_vectors.tocsr().astype(np.float32)
        self.token_vectors = list(token_vectors)
        self.vocab_size = self.pooled.shape[1]

    def search(self, fused_query: sparse.spmatrix, k: int) -> List[RetrievalHit]:
        if fused_query.shape != (1, self.vocab_size):
            raise ValueError(f"query must have shape (1, {self.vocab_size})")
        scores = np.asarray(self.pooled @ fused_query.T).ravel()
        k = min(max(int(k), 0), len(scores))
        if k == 0:
            return []
        # argpartition avoids sorting the whole collection; final order is exact.
        keep = np.argpartition(-scores, k - 1)[:k]
        keep = keep[np.argsort(-scores[keep], kind="stable")]
        return [RetrievalHit(self.doc_ids[i], float(scores[i]), int(i)) for i in keep]


def build_fused_query(query_tokens: Sequence[sparse.spmatrix], alpha: float,
                      vocab_size: int) -> sparse.csr_matrix:

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("interpolation_alpha must be in [0, 1]")
    if not query_tokens:
        return sparse.csr_matrix((1, vocab_size), dtype=np.float32)
    upper = sparse.csr_matrix(sparse.vstack(query_tokens, format="csr").sum(axis=0))
    lower_rows = []
    for token in query_tokens:
        row = token.tocsr()
        if row.nnz:
            max_pos = int(row.indices[np.argmax(row.data)])
            max_val = float(row.data.max())
            lower_rows.append(sparse.csr_matrix(([max_val], ([0], [max_pos])),
                                                shape=(1, vocab_size)))
    lower = (sparse.csr_matrix(sparse.vstack(lower_rows, format="csr").sum(axis=0))
             if lower_rows else sparse.csr_matrix((1, vocab_size)))
    return ((1.0 - alpha) * lower + alpha * upper).tocsr()


def max_pool_tokens(token_vectors: Sequence[sparse.spmatrix], vocab_size: int) -> sparse.csr_matrix:
    if not token_vectors:
        return sparse.csr_matrix((1, vocab_size), dtype=np.float32)
    return sparse.vstack(token_vectors, format="csr").max(axis=0).tocsr()

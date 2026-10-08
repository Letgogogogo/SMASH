"""ColBERT-style MaxSim reranking with a deterministic fallback encoder."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray,
                 max_interaction_tokens: int = 180) -> float:
    q = query_tokens[:max_interaction_tokens]
    d = doc_tokens[:max_interaction_tokens]
    if q.size == 0 or d.size == 0:
        return 0.0
    # Normalized token embeddings make this a scaled cosine similarity.
    sim = q @ d.T
    return float(sim.max(axis=1).sum())


class LateInteractionReranker:
    def __init__(self, encoder, max_interaction_tokens: int = 180):
        self.encoder = encoder
        self.max_interaction_tokens = max_interaction_tokens

    def rerank(self, query: str, candidate_ids: Sequence[str], candidate_texts: Sequence[str]) -> Dict[str, float]:
        q = self.encoder.encode([query])[0]
        docs = self.encoder.encode(candidate_texts)
        return {doc_id: maxsim_score(q, d, self.max_interaction_tokens)
                for doc_id, d in zip(candidate_ids, docs)}

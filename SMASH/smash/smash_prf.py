"""Clustering-driven pseudo-relevance feedback from the SMASH paper."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import List, Sequence

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity


@dataclass
class ClusterFeedback:
    cluster_id: int
    members: np.ndarray
    centroid: sparse.csr_matrix
    weighted_centroid: sparse.csr_matrix
    coherence: float
    accepted: bool
    expansion_indices: List[int]


@dataclass
class PRFResult:
    expanded_query: sparse.csr_matrix
    clusters: List[ClusterFeedback]
    accepted_terms: List[int]
    clustering_prf_ms: float = 0.0
    query_expansion_ms: float = 0.0


class ClusteredPRF:
    def __init__(self, n_clusters: int = 8, top_terms: int = 8,
                 coherence_threshold: float = 0.25, centroid_beta: float = 5.0,
                 random_seed: int = 13):
        self.n_clusters = n_clusters
        self.top_terms = top_terms
        self.coherence_threshold = coherence_threshold
        self.centroid_beta = centroid_beta
        self.random_seed = random_seed

    @staticmethod
    def _confidence(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        shifted = scores - scores.min()
        scale = shifted.max()
        return np.ones_like(scores) if scale < 1e-8 else 0.5 + 0.5 * shifted / scale

    def expand(self, query: sparse.spmatrix, feedback_vectors: sparse.spmatrix,
               feedback_scores: Sequence[float], original_query_text: str = "") -> PRFResult:
        clustering_start = perf_counter_ns()
        q = query.tocsr()
        n, dim = feedback_vectors.shape
        if n == 0:
            return PRFResult(q, [], [], 0.0, 0.0)
        k = max(1, min(self.n_clusters, n))
        # sklearn's KMeans accepts CSR input, avoiding a dense vocab-sized copy.
        labels = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10,
                        max_iter=100).fit_predict(feedback_vectors)
        scores = self._confidence(np.asarray(feedback_scores, dtype=np.float32))
        q_norm = max(float(np.sqrt(q.multiply(q).sum())), 1e-8)
        feedback = feedback_vectors.tocsr()
        result, accepted = [], []
        for cid in range(k):
            members = np.flatnonzero(labels == cid)
            block = feedback[members]
            centroid = sparse.csr_matrix(block.mean(axis=0))
            distances = cosine_distances(block, centroid).ravel()
            weights = np.exp(-self.centroid_beta * distances) * scores[members]
            weights /= max(float(weights.sum()), 1e-8)
            weighted = sparse.csr_matrix(block.multiply(weights[:, None]).sum(axis=0))
            # Intra-cluster cohesion and query/centroid lexical-semantic overlap.
            if len(members) > 1:
                cohesion = float(cosine_similarity(block, block).mean())
            else:
                cohesion = 1.0
            overlap = float(q.multiply(weighted).sum()) / (
                q_norm * max(float(np.sqrt(weighted.multiply(weighted).sum())), 1e-8))
            coherence = 0.5 * cohesion + 0.5 * overlap
            accepted_flag = coherence >= self.coherence_threshold
            if weighted.nnz:
                order = np.argsort(-weighted.data, kind="stable")
                terms = [int(weighted.indices[i]) for i in order[: self.top_terms]]
            else:
                terms = []
            if accepted_flag:
                accepted.extend(terms)
            result.append(ClusterFeedback(cid, members, centroid, weighted,
                                          coherence, accepted_flag, terms))
        clustering_prf_ms = (perf_counter_ns() - clustering_start) / 1_000_000.0

        # Keep the original query weights and add feedback coordinates by max;
        # this prevents expansion from suppressing exact query evidence.
        expansion_start = perf_counter_ns()
        expanded = q.copy().tolil()
        for term in dict.fromkeys(accepted):
            vals = [c.weighted_centroid[0, term] for c in result if c.accepted]
            if vals:
                expanded[0, term] = max(float(expanded[0, term]), float(max(vals)))
        expanded = expanded.tocsr()
        query_expansion_ms = (perf_counter_ns() - expansion_start) / 1_000_000.0
        return PRFResult(
            expanded,
            result,
            list(dict.fromkeys(accepted)),
            clustering_prf_ms,
            query_expansion_ms,
        )

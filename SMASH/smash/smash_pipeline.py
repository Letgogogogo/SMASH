from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Dict, Sequence

import numpy as np
from scipy import sparse

from .config import SmashConfig
from .smash_encoders import (HashingLateInteractionEncoder, HashingSparseEncoder,
                             LateInteractionEncoder, SparseEncoder,
                             TransformerSparseEncoder)
from .smash_index import RetrievalHit, SparseDocumentIndex, build_fused_query
from .smash_prf import ClusteredPRF, PRFResult
from .smash_rerank import LateInteractionReranker


@dataclass
class SearchResult:
    doc_id: str
    sparse_score: float
    rerank_score: float
    stage: str


@dataclass(frozen=True)
class QueryLatency:
    """Wall-clock latency for one query, reported in milliseconds.

    Index construction is intentionally excluded. ``total_ms`` covers the
    online path from query encoding through final result construction.
    """

    first_stage_sparse_retrieval_ms: float
    clustering_prf_ms: float
    query_expansion_ms: float
    second_stage_sparse_retrieval_ms: float
    candidate_preparation_ms: float
    modular_reranking_ms: float
    total_ms: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TimedSearchResult:
    results: list[SearchResult]
    latency: QueryLatency


class SmashRetriever:
    def __init__(self, config: SmashConfig):
        self.config = config
        if config.encoder_backend == "hashing":
            self.sparse_encoder: SparseEncoder = HashingSparseEncoder(
                config.hashing_dim, config.tau, config.sparsity_threshold)
            self.late_encoder = HashingLateInteractionEncoder()
        elif config.encoder_backend == "transformers":
            self.sparse_encoder = TransformerSparseEncoder(
                config.sparse_model_name, config.device, config.max_length,
                config.tau, config.sparsity_threshold, config.allow_model_download)
            self.late_encoder = LateInteractionEncoder(
                config.reranker_model_name, config.device, config.max_length,
                config.allow_model_download)
        else:
            raise ValueError("encoder_backend must be 'transformers' or 'hashing'")
        self.reranker = LateInteractionReranker(self.late_encoder, config.max_interaction_tokens)
        self.prf = ClusteredPRF(config.n_clusters, config.top_expansion_terms,
                                config.coherence_threshold, config.centroid_beta,
                                config.random_seed)
        self.index: SparseDocumentIndex | None = None
        self.last_latency: QueryLatency | None = None

    def build(self, doc_ids: Sequence[str], documents: Sequence[str]) -> None:
        pooled, tokens = self.sparse_encoder.encode(documents)
        self.index = SparseDocumentIndex(doc_ids, documents, pooled, tokens)

    def _require_index(self) -> SparseDocumentIndex:
        if self.index is None:
            raise RuntimeError("build() must be called before search()")
        return self.index

    def search(self, query: str) -> list[SearchResult]:
        """Search while preserving the original result-only API."""
        return self.search_with_latency(query).results

    def search_with_latency(self, query: str) -> TimedSearchResult:
        """Search and return per-stage online latency for one query."""
        index = self._require_index()

        total_start = perf_counter_ns()
        first_stage_start = total_start
        q_pooled, q_tokens = self.sparse_encoder.encode([query])
        fused = build_fused_query(q_tokens[0], self.config.interpolation_alpha, index.vocab_size)
        first = index.search(fused, self.config.first_stage_k)
        first_stage_ms = (perf_counter_ns() - first_stage_start) / 1_000_000.0

        feedback = first[: self.config.feedback_k]
        feedback_vectors = index.pooled[[h.index for h in feedback]] if feedback else sparse.csr_matrix((0, index.vocab_size))
        prf: PRFResult = self.prf.expand(
            q_pooled, feedback_vectors, [h.score for h in feedback], query)
        # The no-PRF ablation keeps exactly the same first-stage fused query;
        # this isolates clustering/expansion rather than changing scoring.
        second_query = fused if self.config.feedback_k == 0 else prf.expanded_query

        second_stage_start = perf_counter_ns()
        second = index.search(second_query, self.config.second_stage_k)
        second_stage_ms = (perf_counter_ns() - second_stage_start) / 1_000_000.0

        candidate_start = perf_counter_ns()
        merged: Dict[str, RetrievalHit] = {h.doc_id: h for h in second}
        for h in first:
            merged.setdefault(h.doc_id, h)
        candidates = list(merged.values())[: self.config.rerank_k]
        candidate_preparation_ms = (perf_counter_ns() - candidate_start) / 1_000_000.0

        reranking_start = perf_counter_ns()
        if not self.config.use_reranker or not candidates:
            results = [SearchResult(h.doc_id, h.score, h.score, "sparse") for h in candidates]
        else:
            ids = [h.doc_id for h in candidates]
            texts = [index.texts[h.index] for h in candidates]
            dense_scores = self.reranker.rerank(query, ids, texts)
            ordered = sorted(candidates, key=lambda h: dense_scores.get(h.doc_id, -np.inf), reverse=True)
            results = [SearchResult(h.doc_id, h.score, dense_scores.get(h.doc_id, 0.0), "reranked") for h in ordered]
        reranking_ms = (perf_counter_ns() - reranking_start) / 1_000_000.0

        total_ms = (perf_counter_ns() - total_start) / 1_000_000.0
        latency = QueryLatency(
            first_stage_sparse_retrieval_ms=first_stage_ms,
            clustering_prf_ms=prf.clustering_prf_ms,
            query_expansion_ms=prf.query_expansion_ms,
            second_stage_sparse_retrieval_ms=second_stage_ms,
            candidate_preparation_ms=candidate_preparation_ms,
            modular_reranking_ms=reranking_ms,
            total_ms=total_ms,
        )
        self.last_latency = latency
        return TimedSearchResult(results, latency)

    def explain(self, query: str) -> dict:
        """Return intermediate stages for the paper's flow-chart/case-study analysis."""
        index = self._require_index()
        q_pooled, q_tokens = self.sparse_encoder.encode([query])
        fused = build_fused_query(q_tokens[0], self.config.interpolation_alpha, index.vocab_size)
        first = index.search(fused, self.config.first_stage_k)
        feedback = first[: self.config.feedback_k]
        vectors = index.pooled[[h.index for h in feedback]] if feedback else sparse.csr_matrix((0, index.vocab_size))
        prf = self.prf.expand(q_pooled, vectors, [h.score for h in feedback], query)
        second_query = fused if self.config.feedback_k == 0 else prf.expanded_query
        second = index.search(second_query, self.config.second_stage_k)
        return {
            "initial_hits": [h.__dict__ for h in first],
            "feedback_hits": [h.__dict__ for h in feedback],
            "clusters": [{"cluster_id": c.cluster_id, "members": c.members.tolist(),
                           "coherence": c.coherence, "accepted": c.accepted,
                           "expansion_indices": c.expansion_indices,
                           "expansion_terms": self.sparse_encoder.terms(c.expansion_indices)}
                          for c in prf.clusters],
            "accepted_expansion_indices": prf.accepted_terms,
            "accepted_expansion_terms": self.sparse_encoder.terms(prf.accepted_terms),
            "expanded_query_nnz": int(prf.expanded_query.nnz),
            "second_stage_hits": [h.__dict__ for h in second],
        }

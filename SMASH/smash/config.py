from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SmashConfig:
    """Runtime configuration corresponding to the knobs described in the paper."""

    sparse_model_name: str = "bert-base-uncased"
    reranker_model_name: str = "colbertv2.0"
    device: str = "cpu"
    max_length: int = 256
    tau: float = 1.0                 # RSG temperature
    sparsity_threshold: float = 0.0    # post-hoc non-negative weight pruning
    interpolation_alpha: float = 0.5   # lower/upper-bound interpolation
    first_stage_k: int = 1000
    feedback_k: int = 10
    n_clusters: int = 8
    top_expansion_terms: int = 8
    coherence_threshold: float = 0.25
    centroid_beta: float = 5.0       # distance-decay sharpness
    second_stage_k: int = 1500
    rerank_k: int = 1500
    max_interaction_tokens: int = 180
    random_seed: int = 13
    use_reranker: bool = True
    allow_model_download: bool = True

    # A small deterministic backend is useful for smoke tests and offline use.
    encoder_backend: str = "transformers"  # transformers | hashing
    hashing_dim: int = 16384

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        if not 0.0 <= self.interpolation_alpha <= 1.0:
            raise ValueError("interpolation_alpha must be in [0, 1]")
        if self.first_stage_k < 0 or self.feedback_k < 0 or self.second_stage_k < 0 or self.rerank_k < 0:
            raise ValueError("retrieval cutoffs must be non-negative")
        if self.n_clusters < 1 or self.top_expansion_terms < 0:
            raise ValueError("n_clusters must be >= 1 and top_expansion_terms non-negative")
        if self.coherence_threshold < 0:
            raise ValueError("coherence_threshold must be non-negative")

    @classmethod
    def demo(cls) -> "SmashConfig":
        return cls(
            encoder_backend="hashing",
            sparse_model_name="hashing-demo",
            reranker_model_name="hashing-demo",
            use_reranker=True,
            max_length=128,
            first_stage_k=100,
            feedback_k=8,
            second_stage_k=100,
            rerank_k=100,
        )

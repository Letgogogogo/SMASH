from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from scipy import sparse


def rsg(logits: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """ReLU-Sigmoid Gating: ReLU(z) * sigmoid(z / tau).

    The output is non-negative, as required by the sparse upper/lower-bound
    retrieval derivation.  ``tau`` must be positive because it controls the
    smoothness of the gate.
    """
    if tau <= 0:
        raise ValueError("RSG temperature tau must be > 0")
    positive = np.maximum(logits, 0.0)
    gate = 1.0 / (1.0 + np.exp(-np.clip(logits / tau, -60.0, 60.0)))
    return positive * gate


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text.lower())


class SparseEncoder:
    vocab_size: int

    def encode(self, texts: Sequence[str], *, return_tokens: bool = True):
        raise NotImplementedError


class HashingSparseEncoder(SparseEncoder):
    """Deterministic lexical sparse encoder for demos and offline smoke tests."""

    def __init__(self, n_features: int = 16384, tau: float = 1.0,
                 threshold: float = 0.0):
        self.vocab_size = int(n_features)
        self.tau = tau
        self.threshold = threshold
        self.id_to_term = {}

    def _term_id(self, term: str) -> int:
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.vocab_size

    def encode(self, texts: Sequence[str], *, return_tokens: bool = True):
        token_vectors = []
        pooled_rows = []
        for text in texts:
            terms = _tokens(text)
            rows = []
            counts = {}
            for term in terms:
                idx = self._term_id(term)
                self.id_to_term.setdefault(idx, term)
                counts[idx] = counts.get(idx, 0.0) + 1.0
            if counts:
                # Positive pseudo-logits followed by the same RSG gate used by
                # the transformer path.  This preserves non-negativity.
                values = {i: float(rsg(np.array([v]), self.tau)[0])
                          for i, v in counts.items()}
                values = {i: v for i, v in values.items() if v > self.threshold}
                rows = list(values.items())
            if rows:
                inds, vals = zip(*rows)
                pooled = sparse.csr_matrix((vals, ([0] * len(inds), inds)),
                                           shape=(1, self.vocab_size))
            else:
                pooled = sparse.csr_matrix((1, self.vocab_size), dtype=np.float32)
            pooled_rows.append(pooled)
            # Hashing tokens are represented as one sparse vector per lexical
            # token, matching the token-level representation needed by PRF.
            tok_rows = []
            for term in terms:
                idx = self._term_id(term)
                tok_rows.append(sparse.csr_matrix(([1.0], ([0], [idx])),
                                                   shape=(1, self.vocab_size)))
            token_vectors.append(tok_rows or [sparse.csr_matrix((1, self.vocab_size))])
        pooled = sparse.vstack(pooled_rows, format="csr") if pooled_rows else sparse.csr_matrix((0, self.vocab_size))
        return pooled, token_vectors

    def terms(self, indices: Iterable[int]) -> List[str]:
        return [self.id_to_term.get(int(i), f"term_{int(i)}") for i in indices]


class TransformerSparseEncoder(SparseEncoder):
    """BERT MLM-head encoder used by the paper's sparse coding stage."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 256,
                 tau: float = 1.0, threshold: float = 0.0,
                 allow_download: bool = True):
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("TransformerSparseEncoder requires torch and transformers") from exc
        self.torch = torch
        kwargs = {} if allow_download else {"local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, **kwargs).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.tau = tau
        self.threshold = threshold
        self.vocab_size = int(self.model.config.vocab_size)
        self._id_to_term = {i: t for t, i in self.tokenizer.get_vocab().items()}

    def encode(self, texts: Sequence[str], *, return_tokens: bool = True):
        torch = self.torch
        batch = self.tokenizer(list(texts), padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**batch).logits.detach().cpu().numpy()
        weights = rsg(logits, self.tau)
        mask = batch["attention_mask"].detach().cpu().numpy().astype(bool)
        pooled_rows, token_vectors = [], []
        for b in range(len(texts)):
            valid = weights[b][mask[b]]
            valid[valid <= self.threshold] = 0.0
            pooled_rows.append(sparse.csr_matrix(valid.max(axis=0, keepdims=True)))
            token_vectors.append([sparse.csr_matrix(row[None, :]) for row in valid])
        pooled = sparse.vstack(pooled_rows, format="csr")
        return pooled, token_vectors

    def terms(self, indices: Iterable[int]) -> List[str]:
        return [self._id_to_term.get(int(i), f"token_{int(i)}") for i in indices]


class LateInteractionEncoder:
    """Dense token encoder for ColBERT-style MaxSim reranking."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 256,
                 allow_download: bool = True):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("LateInteractionEncoder requires torch and transformers") from exc
        kwargs = {} if allow_download else {"local_files_only": True}
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs).to(device).eval()
        self.device, self.max_length = device, max_length

    def encode(self, texts: Sequence[str]) -> List[np.ndarray]:
        torch = self.torch
        batch = self.tokenizer(list(texts), padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        mask = batch["attention_mask"].bool()
        return [hidden[i][mask[i]].cpu().numpy() for i in range(len(texts))]


class HashingLateInteractionEncoder:
    """Small dense token encoder used for deterministic local demos."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> List[np.ndarray]:
        out = []
        for text in texts:
            rows = []
            for term in _tokens(text):
                digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
                seed = int.from_bytes(digest, "little") % (2**32 - 1)
                rng = np.random.default_rng(seed)
                v = rng.normal(size=self.dim).astype(np.float32)
                v /= max(np.linalg.norm(v), 1e-8)
                rows.append(v)
            out.append(np.vstack(rows) if rows else np.zeros((1, self.dim), dtype=np.float32))
        return out

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


def average_precision(ranking: Sequence[str], qrels: Mapping[str, float], relevant_threshold: float = 2.0) -> float:
    relevant = {d for d, r in qrels.items() if r >= relevant_threshold}
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for rank, doc_id in enumerate(ranking, 1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg(ranking: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    def dcg(items):
        return sum((2 ** float(qrels.get(d, 0.0)) - 1) / np.log2(i + 2)
                   for i, d in enumerate(items))
    actual = dcg(ranking[:k])
    ideal = dcg(sorted(qrels, key=qrels.get, reverse=True)[:k])
    return float(actual / ideal) if ideal > 0 else 0.0


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, float], k: int = 1000,
               relevant_threshold: float = 2.0) -> float:
    relevant = {d for d, r in qrels.items() if r >= relevant_threshold}
    return float(len(set(ranking[:k]) & relevant) / len(relevant)) if relevant else 0.0


@dataclass(frozen=True)
class LatencySummary:
    """Aggregate latency statistics over a set of measured queries."""

    query_count: int
    mrt_ms: float
    median_ms: float
    p95_ms: float
    stage_mean_ms: dict[str, float]
    stage_percentage: dict[str, float]
    unaccounted_mean_ms: float
    unaccounted_percentage: float

    def to_dict(self) -> dict:
        return {
            "query_count": self.query_count,
            "mrt_ms": self.mrt_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "stage_mean_ms": self.stage_mean_ms,
            "stage_percentage": self.stage_percentage,
            "unaccounted_mean_ms": self.unaccounted_mean_ms,
            "unaccounted_percentage": self.unaccounted_percentage,
        }


@dataclass(frozen=True)
class WilcoxonResult:
    """Paired Wilcoxon signed-rank comparison of per-query metric scores."""

    query_count: int
    statistic: float
    p_value: float
    alpha: float
    alternative: str
    significant: bool
    system_mean: float
    baseline_mean: float
    mean_difference: float
    wins: int
    ties: int
    losses: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _latency_record_to_mapping(record) -> Mapping[str, float]:
    if isinstance(record, Mapping):
        return record
    if hasattr(record, "to_dict"):
        return record.to_dict()
    raise TypeError("Each latency record must be a mapping or provide to_dict()")


def summarize_latency(
    latency_records: Sequence,
    *,
    total_key: str = "total_ms",
) -> LatencySummary:
    """Compute MRT and the mean percentage of time spent in each stage.

    MRT is the arithmetic mean of end-to-end per-query latency. Index building
    must not be included in the records. Stage percentages use the measured
    end-to-end MRT as the denominator; time not covered by named stages is
    reported as ``unaccounted`` rather than silently redistributed.
    """
    records = [_latency_record_to_mapping(record) for record in latency_records]
    if not records:
        raise ValueError("latency_records must contain at least one query")
    if any(total_key not in record for record in records):
        raise KeyError(f"Every latency record must contain {total_key!r}")

    totals = np.asarray([float(record[total_key]) for record in records], dtype=float)
    if not np.all(np.isfinite(totals)) or np.any(totals < 0):
        raise ValueError("Latency values must be finite and non-negative")

    # Only latency fields are stages. This permits records to carry metadata
    # such as query_id without treating it as a numeric timing value.
    stage_names = sorted(
        key
        for key in set().union(*(record.keys() for record in records))
        if key != total_key and key.endswith("_ms")
    )
    stage_mean_ms = {}
    for stage in stage_names:
        values = np.asarray([float(record.get(stage, 0.0)) for record in records], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Stage {stage!r} contains invalid latency values")
        stage_mean_ms[stage] = float(values.mean())

    mrt_ms = float(totals.mean())
    denominator = mrt_ms if mrt_ms > 0 else 1.0
    stage_percentage = {
        stage: 100.0 * value / denominator
        for stage, value in stage_mean_ms.items()
    }
    accounted = sum(stage_mean_ms.values())
    unaccounted = max(0.0, mrt_ms - accounted)
    return LatencySummary(
        query_count=len(records),
        mrt_ms=mrt_ms,
        median_ms=float(np.median(totals)),
        p95_ms=float(np.percentile(totals, 95)),
        stage_mean_ms=stage_mean_ms,
        stage_percentage=stage_percentage,
        unaccounted_mean_ms=unaccounted,
        unaccounted_percentage=100.0 * unaccounted / denominator,
    )


def wilcoxon_significance(
    system_scores: Mapping[str, float] | Sequence[float],
    baseline_scores: Mapping[str, float] | Sequence[float],
    *,
    alternative: str = "greater",
    alpha: float = 0.05,
    zero_method: str = "wilcox",
) -> WilcoxonResult:
    """Run a paired Wilcoxon signed-rank test on per-query metric scores.

    When mappings are supplied, scores are paired by query ID and mismatched
    query sets are rejected. ``alternative='greater'`` tests whether SMASH is
    better than the baseline; use ``'two-sided'`` for any difference.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    if isinstance(system_scores, Mapping) and isinstance(baseline_scores, Mapping):
        system_ids = set(system_scores)
        baseline_ids = set(baseline_scores)
        if system_ids != baseline_ids:
            missing_system = sorted(baseline_ids - system_ids)
            missing_baseline = sorted(system_ids - baseline_ids)
            raise ValueError(
                "Per-query scores must use identical query IDs; "
                f"missing from system={missing_system}, missing from baseline={missing_baseline}"
            )
        query_ids = sorted(system_ids)
        system = np.asarray([system_scores[qid] for qid in query_ids], dtype=float)
        baseline = np.asarray([baseline_scores[qid] for qid in query_ids], dtype=float)
    elif not isinstance(system_scores, Mapping) and not isinstance(baseline_scores, Mapping):
        system = np.asarray(system_scores, dtype=float)
        baseline = np.asarray(baseline_scores, dtype=float)
    else:
        raise TypeError("system_scores and baseline_scores must both be mappings or both sequences")

    if system.ndim != 1 or baseline.ndim != 1 or len(system) != len(baseline):
        raise ValueError("Paired score arrays must be one-dimensional and equally sized")
    if len(system) == 0:
        raise ValueError("At least one paired query score is required")
    if not np.all(np.isfinite(system)) or not np.all(np.isfinite(baseline)):
        raise ValueError("Scores must be finite")

    differences = system - baseline
    tolerance = 1e-12
    differences[np.abs(differences) <= tolerance] = 0.0
    wins = int(np.sum(differences > tolerance))
    losses = int(np.sum(differences < -tolerance))
    ties = int(len(differences) - wins - losses)

    # scipy raises when every paired difference is zero under wilcox/pratt.
    if np.all(np.abs(differences) <= tolerance):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(
            differences,
            alternative=alternative,
            zero_method=zero_method,
            method="auto",
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)

    return WilcoxonResult(
        query_count=len(system),
        statistic=statistic,
        p_value=p_value,
        alpha=alpha,
        alternative=alternative,
        significant=p_value < alpha,
        system_mean=float(system.mean()),
        baseline_mean=float(baseline.mean()),
        mean_difference=float(differences.mean()),
        wins=wins,
        ties=ties,
        losses=losses,
    )

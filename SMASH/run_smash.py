import argparse
import json
from pathlib import Path

from smash import SmashConfig, SmashRetriever
from smash.smash_evaluation import summarize_latency


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--documents", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--output", default="run.txt")
    p.add_argument("--backend", choices=["hashing", "transformers"], default="hashing")
    p.add_argument("--no-reranker", action="store_true")
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--feedback-k", type=int, default=10)
    p.add_argument(
        "--latency-warmup",
        type=int,
        default=0,
        help="Number of queries to execute before measurement (useful for GPU/model warm-up)",
    )
    p.add_argument(
        "--latency-output",
        help="Optional JSON file containing per-query latency, MRT, and stage percentages",
    )
    args = p.parse_args()
    cfg = SmashConfig.demo() if args.backend == "hashing" else SmashConfig()
    cfg.encoder_backend = args.backend
    cfg.n_clusters = args.n_clusters
    cfg.feedback_k = args.feedback_k
    cfg.use_reranker = not args.no_reranker
    docs, queries = read_jsonl(args.documents), read_jsonl(args.queries)
    if args.latency_warmup < 0:
        p.error("--latency-warmup must be non-negative")
    retriever = SmashRetriever(cfg)
    retriever.build([x["id"] for x in docs], [x["text"] for x in docs])
    for q in queries[: args.latency_warmup]:
        retriever.search(q["text"])
    latency_records = []
    with Path(args.output).open("w", encoding="utf-8") as out:
        for q in queries:
            timed = retriever.search_with_latency(q["text"])
            latency_records.append({"query_id": q["id"], **timed.latency.to_dict()})
            for rank, result in enumerate(timed.results, 1):
                out.write(f"{q['id']} Q0 {result.doc_id} {rank} {result.rerank_score:.8f} smash\n")
    if args.latency_output:
        report = {
            "per_query": latency_records,
            "summary": summarize_latency(latency_records).to_dict(),
        }
        with Path(args.latency_output).open("w", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

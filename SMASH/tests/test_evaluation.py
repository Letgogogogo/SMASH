import unittest

from smash.smash_evaluation import summarize_latency, wilcoxon_significance


class LatencyEvaluationTests(unittest.TestCase):
    def test_mrt_and_stage_percentages(self):
        summary = summarize_latency([
            {
                "query_id": "q1",
                "first_stage_sparse_retrieval_ms": 20.0,
                "clustering_prf_ms": 50.0,
                "query_expansion_ms": 5.0,
                "second_stage_sparse_retrieval_ms": 10.0,
                "candidate_preparation_ms": 2.0,
                "modular_reranking_ms": 10.0,
                "total_ms": 100.0,
            },
            {
                "query_id": "q2",
                "first_stage_sparse_retrieval_ms": 40.0,
                "clustering_prf_ms": 100.0,
                "query_expansion_ms": 10.0,
                "second_stage_sparse_retrieval_ms": 20.0,
                "candidate_preparation_ms": 4.0,
                "modular_reranking_ms": 20.0,
                "total_ms": 200.0,
            },
        ])
        self.assertEqual(summary.query_count, 2)
        self.assertAlmostEqual(summary.mrt_ms, 150.0)
        self.assertAlmostEqual(summary.stage_mean_ms["clustering_prf_ms"], 75.0)
        self.assertAlmostEqual(summary.stage_percentage["clustering_prf_ms"], 50.0)
        self.assertAlmostEqual(summary.unaccounted_mean_ms, 4.5)
        self.assertAlmostEqual(summary.unaccounted_percentage, 3.0)

    def test_rejects_empty_latency_input(self):
        with self.assertRaises(ValueError):
            summarize_latency([])


class WilcoxonEvaluationTests(unittest.TestCase):
    def test_pairs_mapping_scores_by_query_id(self):
        result = wilcoxon_significance(
            {"q2": 0.7, "q1": 0.8, "q3": 0.9, "q4": 0.6},
            {"q1": 0.5, "q2": 0.4, "q3": 0.6, "q4": 0.3},
            alternative="greater",
        )
        self.assertEqual(result.query_count, 4)
        self.assertEqual(result.wins, 4)
        self.assertEqual(result.ties, 0)
        self.assertEqual(result.losses, 0)
        self.assertGreater(result.mean_difference, 0.0)

    def test_all_ties_return_p_one(self):
        result = wilcoxon_significance([0.5, 0.5], [0.5, 0.5])
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.significant)
        self.assertEqual(result.ties, 2)

    def test_rejects_mismatched_query_ids(self):
        with self.assertRaises(ValueError):
            wilcoxon_significance({"q1": 1.0}, {"q2": 1.0})


if __name__ == "__main__":
    unittest.main()

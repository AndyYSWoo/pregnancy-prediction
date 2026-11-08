import unittest
from types import SimpleNamespace

import numpy as np

from data_gen import (
    cycleLength,
    genCycleData,
    symptomNames,
    convertToNdarray,
)
from graph_imputation import (
    N_NODES,
    hide_events,
    run_experiment,
    run_multiseed_evaluation,
    _recall_at_hidden_k,
)
from graph_autoencoder import BipartiteGraphAutoencoder, build_patient_feature_edges


class GraphImputationTests(unittest.TestCase):
    def test_generator_is_seeded_and_has_six_categories(self):
        first = genCycleData(4, seed=7)
        second = genCycleData(4, seed=7)
        np.testing.assert_array_equal(
            np.stack([events for events, _ in first]),
            np.stack([events for events, _ in second]),
        )
        self.assertEqual(len(symptomNames), 6)
        self.assertEqual(convertToNdarray(first).shape, (4, 1 + cycleLength * 6))

    def test_patient_traits_are_reproducible_and_change_within_patient_patterns(self):
        first = genCycleData(500, seed=17, structure_strength=1.2, return_traits=True)
        second = genCycleData(500, seed=17, structure_strength=1.2, return_traits=True)
        np.testing.assert_array_equal(
            np.stack([events for events, _, _ in first]),
            np.stack([events for events, _, _ in second]),
        )
        self.assertEqual(
            [traits["sex_activity"] for _, _, traits in first],
            [traits["sex_activity"] for _, _, traits in second],
        )
        sex_activity = np.array([traits["sex_activity"] for _, _, traits in first])
        sex_counts = np.array([events[:, :3].sum() for events, _, _ in first])
        high_group = sex_counts[sex_activity > np.quantile(sex_activity, 0.75)]
        low_group = sex_counts[sex_activity < np.quantile(sex_activity, 0.25)]
        self.assertGreater(high_group.mean(), low_group.mean())

    def test_zero_structure_strength_removes_latent_event_propensity(self):
        data = genCycleData(300, seed=21, structure_strength=0)
        event_totals = np.array([events.sum() for events, _ in data])
        self.assertGreater(event_totals.std(), 0)
        zero_a = genCycleData(10, seed=21, structure_strength=0)
        zero_b = genCycleData(10, seed=21, structure_strength=0)
        np.testing.assert_array_equal(
            np.stack([events for events, _ in zero_a]),
            np.stack([events for events, _ in zero_b]),
        )
        self.assertEqual([label for _, label in zero_a], [label for _, label in zero_b])

    def test_day_dropout_hides_all_six_values_including_zeros(self):
        events = np.stack([row for row, _ in genCycleData(4, seed=3)])
        observed, hidden = hide_events(events, 0.5, np.random.default_rng(3))
        grouped = hidden.reshape(-1, cycleLength, len(symptomNames))
        for day_mask in grouped:
            self.assertTrue(np.all(day_mask == day_mask[:, :1]))
        self.assertTrue(np.any(hidden & (events == 0)))
        np.testing.assert_array_equal(observed[~hidden], events[~hidden])
        self.assertTrue(np.all(observed[hidden] == 0))

    def test_positive_event_dropout_remains_optional(self):
        events = np.stack([row for row, _ in genCycleData(4, seed=5)])
        _, hidden = hide_events(
            events, 0.8, np.random.default_rng(5), pattern="positive_event"
        )
        self.assertTrue(np.all(events[hidden] == 1))

    def test_hidden_recall_ranks_only_hidden_candidates(self):
        truth = np.zeros((1, N_NODES))
        hidden = np.zeros((1, N_NODES), dtype=bool)
        scores = np.zeros((1, N_NODES))
        hidden[0, :3] = True
        truth[0, 1] = 1
        scores[0, 50] = 100  # observed values must not enter the ranking set
        scores[0, 1] = 2
        self.assertEqual(_recall_at_hidden_k(scores, hidden, truth), 1.0)

    def test_autoencoder_training_ignores_masked_ground_truth(self):
        events = np.stack([row for row, _ in genCycleData(12, seed=13)])
        observed, hidden = hide_events(events, 0.5, np.random.default_rng(13))
        mask = ~hidden
        changed_hidden_truth = events.copy()
        changed_hidden_truth[hidden] = 1 - changed_hidden_truth[hidden]
        first = BipartiteGraphAutoencoder(len(events), seed=13).fit(
            events, observation_mask=mask, epochs=4
        )
        second = BipartiteGraphAutoencoder(len(events), seed=13).fit(
            changed_hidden_truth, observation_mask=mask, epochs=4
        )
        np.testing.assert_allclose(first.patient_embeddings, second.patient_embeddings)
        np.testing.assert_allclose(first.day_embeddings, second.day_embeddings)
        np.testing.assert_allclose(first.category_embeddings, second.category_embeddings)
        np.testing.assert_allclose(first.feature_bias, second.feature_bias)

    def test_observed_patient_feature_edges_drive_message_passing(self):
        events = np.zeros((2, N_NODES))
        observed = np.ones_like(events, dtype=bool)
        events[0, 0] = 1
        events[1, 1] = 1
        heldout = np.zeros_like(observed)
        heldout[0, 0] = True
        edges = build_patient_feature_edges(events, observed, heldout)
        self.assertEqual(edges.sum(), 1)
        self.assertEqual(edges[0, 0], 0)
        model = BipartiteGraphAutoencoder(2, seed=3)
        before_patient, before_feature = model._encode_fit_graph(edges)
        added_edge = edges.copy()
        added_edge[0, 1] = 1
        after_patient, after_feature = model._encode_fit_graph(added_edge)
        self.assertFalse(np.allclose(before_patient, after_patient))
        self.assertFalse(np.allclose(before_feature, after_feature))

    def test_patient_neighbor_evidence_changes_imputation(self):
        fit_events = np.zeros((3, N_NODES))
        fit_events[:, 0] = 1
        fit_events[0, 1] = 1
        fit_mask = np.ones_like(fit_events, dtype=bool)
        model = BipartiteGraphAutoencoder(3, seed=9).fit(
            fit_events, observation_mask=fit_mask, epochs=8
        )
        query = np.zeros((1, N_NODES))
        query_mask = np.ones_like(query, dtype=bool)
        query[0, 0] = 1
        query_mask[0, 1] = False
        first = model.predict(query, query_mask, fit_events, fit_mask)
        query[0, 2] = 1
        second = model.predict(query, query_mask, fit_events, fit_mask)
        self.assertFalse(np.allclose(first, second))

    def test_graph_fit_and_end_to_end_metrics(self):
        cycles = genCycleData(30, seed=4)
        events = np.stack([row for row, _ in cycles])
        model = BipartiteGraphAutoencoder(len(events), seed=4).fit(events, epochs=3)
        predictions = model.predict(events[:5], np.ones_like(events[:5], dtype=bool), events, np.ones_like(events, dtype=bool))
        self.assertEqual(predictions.shape, (5, N_NODES))
        self.assertTrue(np.all(np.isfinite(predictions)))
        result = run_experiment(
            num_patients=60, missing_rate=0.4, epochs=3, seed=4,
            run_downstream=False,
        )
        self.assertEqual(result["pregnancy_prediction"]["status"], "skipped")
        self.assertEqual(
            result["pregnancy_prediction"]["entry_point"],
            "main.py --model=LSTM -> models/LSTMModel.py",
        )
        self.assertIn("graph_recall_at_hidden_k", result["imputation"])
        self.assertIn("hidden_binary_reconstruction", result["imputation"])
        self.assertGreater(result["patient_graph"]["fit_patient_feature_edges"], 0)

    def test_repo_model_runner_uses_existing_main_entry_point(self):
        from repo_model_runner import run_repo_model_comparison

        calls = []

        def fake_runner(command, **kwargs):
            calls.append(command)
            output = (
                "For whole test set: auc 0.750000 1-cycle P_preg (0.5, 0.2)"
                if "--mode=test" in command else ""
            )
            return SimpleNamespace(returncode=0, stderr="", stdout=output)

        features = np.zeros((4, N_NODES))
        results = run_repo_model_comparison(
            {"zero_fill": (features[:3], features[:1], features[:1])},
            np.array([0, 1, 0]), np.array([1]), np.array([0]),
            model="LSTM", epochs=1, runner=fake_runner,
        )
        self.assertEqual(results["zero_fill"]["auc"], 0.75)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1].endswith("main.py"))
        self.assertIn("--model=LSTM", calls[0])
        self.assertIn("--mode=train", calls[0])
        self.assertIn("--mode=test", calls[1])

    def test_multiseed_report_aggregates_each_baseline(self):
        report = run_multiseed_evaluation(
            num_patients=40,
            missing_rate=0.25,
            epochs=2,
            seeds=(2, 3),
            structure_strengths=(0.8,),
            run_downstream=False,
        )
        self.assertEqual(len(report["runs"]), 2)
        summary = report["summaries"][0]
        self.assertEqual(summary["seed_count"], 2)
        self.assertEqual(
            summary["imputation_metrics"]["imputation.graph_recall_at_hidden_k"]["n"],
            2,
        )
        self.assertIsNone(
            summary["downstream_model_auc"]["graph_imputation"]["mean_auc"]
        )


if __name__ == "__main__":
    unittest.main()

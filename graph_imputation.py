#!/usr/bin/env python3
"""Numpy-only temporal bipartite graph autoencoder and evaluation workflow."""
import argparse
import importlib.util
import json
import sys

import numpy as np

from data_gen import (
    defaultPatientStructureStrength,
    genCycleData,
    symptomNames,
    cycleLength,
)
from graph_autoencoder import BipartiteGraphAutoencoder
from repo_model_runner import RepoModelUnavailable, run_repo_model_comparison


N_FEATURES = len(symptomNames)
N_NODES = cycleLength * N_FEATURES


def hide_events(events, rate, rng, pattern="day"):
    """Hide complete daily reports, or optionally individual positive events."""
    events = np.asarray(events, dtype=np.float64)
    if not 0 <= rate < 1:
        raise ValueError("missing rate must be in [0, 1)")
    original_shape = events.shape
    if events.ndim == 3 and events.shape[1:] == (cycleLength, N_FEATURES):
        events = events.reshape(len(events), N_NODES)
    elif events.ndim != 2 or events.shape[1] != N_NODES:
        raise ValueError("events must have shape (patients, 144) or (patients, 24, 6)")
    if pattern == "day":
        missing_days = rng.random((len(events), cycleLength)) < rate
        hidden = np.repeat(missing_days, N_FEATURES, axis=1)
    elif pattern == "positive_event":
        hidden = (events > 0) & (rng.random(events.shape) < rate)
    else:
        raise ValueError("pattern must be 'day' or 'positive_event'")
    return (events * ~hidden).reshape(original_shape), hidden.reshape(original_shape)


def auc(labels, scores):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _recall_at_hidden_k(scores, hidden, truth):
    hidden = np.asarray(hidden, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    total = int((hidden & truth).sum())
    if not total:
        return None
    recovered = 0
    for row in range(hidden.shape[0]):
        count = int((hidden[row] & truth[row]).sum())
        if count:
            candidates = np.flatnonzero(hidden[row])
            k = min(count, len(candidates))
            chosen = candidates[np.argpartition(scores[row, candidates], -k)[-k:]]
            recovered += int((hidden[row, chosen] & truth[row, chosen]).sum())
    return recovered / total


def _hidden_binary_metrics(scores, hidden, truth):
    selected = np.asarray(hidden, dtype=bool).reshape(-1)
    if not selected.any():
        return {"count": 0, "accuracy": None, "precision": None, "recall": None, "auc": None}
    actual = np.asarray(truth, dtype=bool).reshape(-1)[selected]
    predicted_scores = np.asarray(scores).reshape(-1)[selected]
    predicted = predicted_scores >= 0.5
    true_positive = int((predicted & actual).sum())
    return {
        "count": int(selected.sum()),
        "accuracy": float((predicted == actual).mean()),
        "precision": true_positive / int(predicted.sum()) if predicted.any() else 0.0,
        "recall": true_positive / int(actual.sum()) if actual.any() else None,
        "auc": auc(actual, predicted_scores),
    }


def run_multiseed_evaluation(
    num_patients=600, missing_rate=0.3, epochs=80, seeds=(1, 2, 3),
    structure_strengths=(0.0, 0.8, 1.5), prediction_model="LSTM",
    model_epochs=20, run_downstream=True,
):
    """Run each structural condition over fixed seeds and aggregate key metrics."""
    if not seeds or not structure_strengths:
        raise ValueError("multi-seed evaluation requires seeds and structure strengths")
    runs = []
    for strength in structure_strengths:
        for seed in seeds:
            result = run_experiment(
                num_patients=num_patients,
                missing_rate=missing_rate,
                epochs=epochs,
                seed=seed,
                prediction_model=prediction_model,
                model_epochs=model_epochs,
                run_downstream=run_downstream,
                structure_strength=strength,
            )
            runs.append({
                "seed": seed,
                "structure_strength": strength,
                **result,
            })

    summaries = []
    metric_paths = (
        ("imputation", "graph_recall_at_hidden_k"),
        ("imputation", "prevalence_recall_at_hidden_k"),
        ("imputation", "hidden_binary_reconstruction", "auc"),
        ("imputation", "hidden_binary_reconstruction", "accuracy"),
    )
    for strength in structure_strengths:
        selected = [run for run in runs if run["structure_strength"] == strength]
        metrics = {}
        for path in metric_paths:
            values = []
            for run in selected:
                value = run
                for key in path:
                    value = value[key]
                if value is not None:
                    values.append(float(value))
            metrics[".".join(path)] = {
                "n": len(values),
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else (
                    0.0 if values else None
                ),
            }
        prediction_conditions = {}
        for condition in (
            "zero_fill", "prevalence_fill", "graph_imputation",
            "fully_observed_reference",
        ):
            values = [
                run["pregnancy_prediction"].get("conditions", {}).get(condition, {}).get("auc")
                for run in selected
            ]
            values = [float(value) for value in values if value is not None]
            prediction_conditions[condition] = {
                "n": len(values),
                "mean_auc": float(np.mean(values)) if values else None,
                "std_auc": float(np.std(values, ddof=1)) if len(values) > 1 else (
                    0.0 if values else None
                ),
            }
        summaries.append({
            "structure_strength": strength,
            "seed_count": len(selected),
            "imputation_metrics": metrics,
            "downstream_model_auc": prediction_conditions,
            "downstream_statuses": sorted(set(
                run["pregnancy_prediction"]["status"] for run in selected
            )),
        })
    return {
        "evaluation": {
            "model": prediction_model,
            "num_patients": num_patients,
            "missing_rate": missing_rate,
            "imputer_epochs": epochs,
            "downstream_epochs": model_epochs,
            "seeds": list(seeds),
            "structure_strengths": list(structure_strengths),
            "aggregate": "mean and sample standard deviation across seeds (population std for a single seed)",
        },
        "summaries": summaries,
        "runs": runs,
    }


def run_experiment(
    num_patients=600, missing_rate=0.3, epochs=80, seed=1,
    prediction_model="LSTM", model_epochs=20, run_downstream=True,
    structure_strength=defaultPatientStructureStrength,
):
    rng = np.random.default_rng(seed)
    cycles = genCycleData(
        num_patients, seed=seed, structure_strength=structure_strength
    )
    truth = np.stack([cycle for cycle, _ in cycles]).reshape(num_patients, N_NODES)
    labels = np.array([label for _, label in cycles])
    indices = rng.permutation(num_patients)
    train_idx, test_idx = indices[:int(0.7 * num_patients)], indices[int(0.7 * num_patients):]
    dev_count = max(1, int(0.1 * len(train_idx)))
    fit_idx, dev_idx = train_idx[:-dev_count], train_idx[-dev_count:]
    fit_local_idx = np.arange(len(train_idx) - dev_count)
    dev_local_idx = np.arange(len(train_idx) - dev_count, len(train_idx))
    train_truth, test_truth = truth[train_idx], truth[test_idx]

    train_observed, train_hidden = hide_events(train_truth, missing_rate, rng, pattern="day")
    test_observed, test_hidden = hide_events(test_truth, missing_rate, rng, pattern="day")
    train_known = ~train_hidden
    test_known = ~test_hidden
    fit_observed = train_observed[fit_local_idx]
    fit_known = train_known[fit_local_idx]
    dev_observed, dev_hidden = train_observed[dev_local_idx], train_hidden[dev_local_idx]
    dev_known = ~dev_hidden
    model = BipartiteGraphAutoencoder(
        patient_count=len(fit_observed), seed=seed
    ).fit(
        fit_observed,
        observation_mask=fit_known,
        epochs=epochs,
        validation_events=dev_observed,
        validation_mask=dev_known,
    )
    prevalence_denominator = np.maximum(fit_known.sum(axis=0), 1)
    prevalence = fit_observed.sum(axis=0) / prevalence_denominator
    prevalence_scores = np.broadcast_to(prevalence, test_truth.shape)
    fit_probabilities, train_neighbor_counts = model.predict(
        fit_observed, fit_known, fit_observed, fit_known,
        exclude_self=True, return_neighbor_counts=True,
    )
    dev_probabilities, dev_neighbor_counts = model.predict(
        dev_observed, dev_known, fit_observed, fit_known,
        return_neighbor_counts=True,
    )
    test_scores, test_neighbor_counts = model.predict(
        test_observed, test_known, fit_observed, fit_known,
        return_neighbor_counts=True,
    )
    hidden_positive = test_hidden & (test_truth > 0)
    imputation = {
        "hidden_positive_count": int(hidden_positive.sum()),
        "hidden_binary_reconstruction": _hidden_binary_metrics(
            test_scores, test_hidden, test_truth
        ),
        "graph_recall_at_hidden_k": _recall_at_hidden_k(test_scores, test_hidden, test_truth),
        "prevalence_recall_at_hidden_k": _recall_at_hidden_k(prevalence_scores, test_hidden, test_truth),
    }
    fit_imputed = np.where(fit_known, fit_observed, fit_probabilities)
    dev_imputed = np.where(dev_known, dev_observed, dev_probabilities)
    test_imputed = np.where(test_known, test_observed, test_scores)
    prevalence_imputed_train = np.where(fit_known, fit_observed, prevalence)
    prevalence_imputed_dev = np.where(dev_known, dev_observed, prevalence)
    prevalence_imputed_test = np.where(test_known, test_observed, prevalence)

    prediction = {
        "status": "skipped" if not run_downstream else "unavailable",
        "model": prediction_model,
        "entry_point": "main.py --model={} -> models/{}Model.py".format(
            prediction_model, prediction_model
        ),
        "runtime": {
            "python": sys.version.split()[0],
            "tensorflow_installed": importlib.util.find_spec("tensorflow") is not None,
        },
    }
    if run_downstream:
        conditions = {
            "zero_fill": (
                fit_observed,
                dev_observed,
                test_observed,
            ),
            "prevalence_fill": (
                prevalence_imputed_train,
                prevalence_imputed_dev,
                prevalence_imputed_test,
            ),
            "graph_imputation": (
                fit_imputed,
                dev_imputed,
                test_imputed,
            ),
            "fully_observed_reference": (
                train_truth[fit_local_idx],
                train_truth[dev_local_idx],
                test_truth,
            ),
        }
        try:
            results = run_repo_model_comparison(
                conditions, labels[fit_idx], labels[dev_idx], labels[test_idx],
                model=prediction_model, epochs=model_epochs,
            )
            prediction.update({"status": "complete", "conditions": results})
        except RepoModelUnavailable as error:
            prediction.update({"status": "unavailable", "reason": str(error)})
    return {
        "assumptions": {
            "dataset": "synthetic 24-day cycles only; not mcPHASES",
            "patient_structure": "stable within-cycle activity, reporting/category preferences, and fertility susceptibility; structure strength scales latent effects",
            "missingness": "MCAR day-level reporting dropout by default; all six binary values on a dropped day, including zeros, are unobserved",
            "graph": "learned bipartite graph over fit patient nodes and day-conditioned feature nodes, with observed positive values as edges and normalized message passing; new patients aggregate visible feature messages and similar fit-patient states",
            "imputer_training": "self-supervised reconstruction of randomly hidden training events; pregnancy labels are not used",
            "patient_graph": "learned patient and time-conditioned (day, feature) nodes exchange messages over observed positive patient-feature edges; dev and test patients connect only to learned fit-patient nodes; no labels enter graph construction",
        },
        "missing_rate": missing_rate,
        "missingness_pattern": "day",
        "patient_structure_strength": structure_strength,
        "patient_graph": {
            "fit_patient_feature_edges": int(np.sum((fit_observed > 0) & fit_known)),
            "fit_neighbor_count_mean": float(train_neighbor_counts.mean()),
            "dev_neighbor_count_mean": float(dev_neighbor_counts.mean()),
            "test_neighbor_count_mean": float(test_neighbor_counts.mean()),
        },
        "imputation": imputation,
        "imputer_training": {
            "epochs_trained": model.epochs_trained,
            "best_dev_observed_bce": model.best_validation_loss,
        },
        "pregnancy_prediction": prediction,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=600)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prediction-model", choices=("LSTM", "LR", "BMS"), default="LSTM")
    parser.add_argument("--model-epochs", type=int, default=20)
    parser.add_argument("--sweep", action="store_true", help="run all seed/structure combinations")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--structure-strengths", default="0,0.8,1.5")
    parser.add_argument(
        "--structure-strength", type=float,
        default=defaultPatientStructureStrength,
        help="strength of stable synthetic patient propensities, from 0 to 2",
    )
    parser.add_argument("--skip-downstream", action="store_true")
    args = parser.parse_args()
    if args.patients < 4:
        parser.error("--patients must be at least 4")
    if args.model_epochs < 1:
        parser.error("--model-epochs must be positive")
    if args.sweep:
        try:
            seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
            strengths = [
                float(value) for value in args.structure_strengths.split(",")
                if value.strip()
            ]
        except ValueError:
            parser.error("--seeds and --structure-strengths must be comma-separated numbers")
        print(json.dumps(run_multiseed_evaluation(
            num_patients=args.patients,
            missing_rate=args.missing_rate,
            epochs=args.epochs,
            seeds=seeds,
            structure_strengths=strengths,
            prediction_model=args.prediction_model,
            model_epochs=args.model_epochs,
            run_downstream=not args.skip_downstream,
        ), indent=2))
    else:
        print(json.dumps(run_experiment(
            args.patients, args.missing_rate, args.epochs, args.seed,
            prediction_model=args.prediction_model, model_epochs=args.model_epochs,
            run_downstream=not args.skip_downstream,
            structure_strength=args.structure_strength,
        ), indent=2))


if __name__ == "__main__":
    main()

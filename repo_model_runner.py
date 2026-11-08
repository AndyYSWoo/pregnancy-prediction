"""Run the repository's TensorFlow models on prepared train/dev/test arrays."""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np


_TEST_AUC = re.compile(r"For whole test set: auc ([0-9.eE+-]+)")
_REPOSITORY_ROOT = os.path.dirname(os.path.abspath(__file__))


class RepoModelUnavailable(RuntimeError):
    """The legacy repository model entry point could not be executed."""


def _save_split(path, features, labels):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1, 1)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have matching patient counts")
    np.save(path, np.concatenate((labels, features), axis=1))


def _main_command(model, mode, data_dir, experiment_dir, epochs, batch_size):
    main_path = os.path.join(_REPOSITORY_ROOT, "main.py")
    return [
        sys.executable, main_path,
        "--model=" + model,
        "--mode=" + mode,
        "--per=cycle",
        "--training_data_dir=" + os.path.join(data_dir, "train"),
        "--dev_data_file=" + os.path.join(data_dir, "dev", "dev.npy"),
        "--test_data_file=" + os.path.join(data_dir, "test", "test.npy"),
        "--tb_dir=" + experiment_dir,
        "--train_dir=" + os.path.join(experiment_dir, "run"),
        "--retrain=True",
        "--num_epochs=" + str(epochs),
        "--batch_size=" + str(batch_size),
        "--eval_every=1",
        "--max_cycle_length=24",
        "--num_daily_feature=6",
        "--lstm_layer=1",
        "--lstm_size=32",
        "--fc_layer=1",
        "--fc_size=32",
        "--dropout=1.0",
    ]


def run_repo_model_comparison(
    conditions, train_labels, dev_labels, test_labels, model="LSTM",
    epochs=20, batch_size=64, runner=subprocess.run,
):
    """Fit one of the repo's models independently for each input condition."""
    if model not in ("LSTM", "LR", "BMS"):
        raise ValueError("model must be one of LSTM, LR, or BMS")
    results = {}
    with tempfile.TemporaryDirectory(prefix="pregnancy-prediction-models-") as root:
        for condition, (train_x, dev_x, test_x) in conditions.items():
            data_dir = os.path.join(root, condition, "data")
            experiment_dir = os.path.join(root, condition, "experiments")
            train_dir = os.path.join(data_dir, "train")
            dev_dir = os.path.join(data_dir, "dev")
            test_dir = os.path.join(data_dir, "test")
            for directory in (train_dir, dev_dir, test_dir, experiment_dir):
                os.makedirs(directory, exist_ok=True)
            _save_split(os.path.join(train_dir, "train.npy"), train_x, train_labels)
            _save_split(os.path.join(dev_dir, "dev.npy"), dev_x, dev_labels)
            _save_split(os.path.join(test_dir, "test.npy"), test_x, test_labels)

            environment = os.environ.copy()
            old_path = environment.get("PYTHONPATH", "")
            model_path = os.path.join(_REPOSITORY_ROOT, "models")
            environment["PYTHONPATH"] = os.pathsep.join(
                path for path in (model_path, _REPOSITORY_ROOT, old_path) if path
            )
            for mode in ("train", "test"):
                command = _main_command(
                    model, mode, data_dir, experiment_dir, epochs, batch_size
                )
                try:
                    completed = runner(
                        command, cwd=root, env=environment, text=True,
                        capture_output=True, check=False,
                    )
                except OSError as error:
                    raise RepoModelUnavailable(
                        "Could not start repository main.py for model {}: {}".format(model, error)
                    )
                if completed.returncode:
                    detail = (completed.stderr or completed.stdout).strip()
                    raise RepoModelUnavailable(
                        "Repository model {} failed in {} mode (exit {}): {}".format(
                            model, mode, completed.returncode, detail[-1200:]
                        )
                    )
                if mode == "test":
                    match = _TEST_AUC.search(completed.stdout)
                    if not match:
                        raise RepoModelUnavailable(
                            "Repository model {} did not emit its expected test AUC; output: {}".format(
                                model, completed.stdout[-1200:]
                            )
                        )
                    results[condition] = {"auc": float(match.group(1))}
    return results

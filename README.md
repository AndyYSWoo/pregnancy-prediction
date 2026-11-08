# Pregnancy Prediction
This repo provides source code for paper: [Predicting pregnancy using large-scale data from a women's health tracking mobile application](https://arxiv.org/abs/1812.02222).

## File Descriptions
`main.py`: main entry of models

`data_gen.py`: a synthetic data generator

`demo_on_how_to_run_models.ipynb`: notebook to illustrate how to generate data and run models

`models/`: implementation of different models mentioned in the paper

## Synthetic missing-data and graph-imputation prototype

The original TensorFlow models above are retained. For a reproducible prototype
that runs independently of that legacy stack, install Python 3 and NumPy:

```sh
python3 -m pip install -r requirements-prototype.txt
python3 graph_imputation.py --sweep --patients 600 --missing-rate 0.3 \
  --epochs 80 --model-epochs 20 --seeds 1,2,3 \
  --structure-strengths 0,0.8,1.5
python3 -m unittest discover -s tests
```

The generator keeps the six existing binary daily categories and pregnancy
label. The experiment splits patients before fitting, then independently hides
a configurable fraction of **whole daily reports** by default. When a report
is missing, all six binary values for that day—including zeros—are unknown.
Ground truth is retained separately from the observation mask, so observed
zeros remain distinct from unreported values. The `hide_events` helper also
supports optional `positive_event` masking for event-level dropout experiments.

Each generated patient receives stable latent propensities for sex activity,
emotion-reporting frequency, feature-category preferences, and fertility
susceptibility. Those traits affect daily event rates/categories and the
pregnancy label risk across that patient's cycle. `--structure-strength`
controls their effect from `0` (no latent effect) through `2` (stronger
heterogeneity); the default is `0.8`. This is a deliberately simplified
synthetic assumption: traits are independently sampled, fixed within one
patient-cycle, and are not intended as clinical factors. Generation is seeded,
and `genCycleData(..., return_traits=True)` exposes the sampled traits for
analysis.

The imputer is a learned, self-supervised bipartite graph autoencoder. It
creates one learned node per fit patient and one time-conditioned node per
(day, feature-category) pair; each feature node is the sum of a learned shared
day embedding and category embedding. Observed positive patient-feature values
form graph edges. Patient and feature nodes exchange normalized messages over
these bipartite edges, and a learned inner-product decoder reconstructs
artificially withheld positive values plus sampled observed zeros using binary
cross-entropy. Unknown values never enter message passing or the loss. Dev
observations select the training epoch by observed-entry BCE; dev values do
not update parameters. At inference, new patient nodes combine their visible
event-edge messages with learned states from similar fit-patient neighbors.
Fit patients use leave-one-out neighbors, while dev and test patients connect
only to fit patients. No pregnancy labels enter graph construction or imputer
training.

Evaluation reports hidden-positive recall@K, ranking only hidden cells, and
accuracy, precision, recall, and AUC over all withheld binary values. Imputation
baselines include fit-partition prevalence and zero fill; downstream
classification compares zero-fill, prevalence-fill, graph-imputed, and
fully-observed reference inputs using the repository's actual `main.py`
`--model=LSTM` path. The full command runs a multi-seed sweep (default strengths
0, 0.8, 1.5 and seeds 1, 2, 3), printing every run and per-strength mean/sample
standard deviation for reconstruction metrics and held-out LSTM AUC. Use
`--prediction-model LR` or `--prediction-model BMS` for other repo models.
To reduce runtime while testing the sweep, set smaller `--patients`,
`--epochs`, and `--model-epochs`; `--skip-downstream` omits the LSTM jobs.

This prototype simulates missing completely at random at the day-report level.
It does not model real reporting behavior, uncertainty in event recording,
patient metadata, or the full mcPHASES feature schema. Synthetic results are
only a pipeline sanity check and do **not** establish imputation or pregnancy
prediction performance on mcPHASES. The current workflow generates synthetic
cycles in memory and does not download or require mcPHASES.

### Example multi-seed report

The following report was produced with 600 synthetic patients per run, 30%
whole-day MCAR dropout, 40 GAE epochs maximum, 10 downstream LSTM epochs, and
seeds 11, 23, and 37 at each structure strength. Values are mean ± sample
standard deviation across the three held-out test runs. Recall@K ranks only
withheld candidate cells, with K equal to the number of hidden positive cells;
binary reconstruction AUC/accuracy use all withheld cells. The downstream
column order is zero-fill / prevalence-fill / GAE-imputed / fully observed.

| Structure strength | GAE recall@K | Prevalence recall@K | GAE hidden-cell AUC | GAE hidden-cell accuracy |
|---:|---:|---:|---:|---:|
| 0.0 | 0.108 ± 0.011 | 0.112 ± 0.011 | 0.529 ± 0.011 | 0.882 ± 0.032 |
| 0.8 | 0.200 ± 0.016 | 0.147 ± 0.010 | 0.611 ± 0.016 | 0.859 ± 0.044 |
| 1.5 | 0.330 ± 0.021 | 0.169 ± 0.007 | 0.711 ± 0.016 | 0.850 ± 0.035 |

| Structure strength | LSTM test AUC: zero / prevalence / GAE / fully observed |
|---:|---:|
| 0.0 | 0.747 ± 0.021 / 0.741 ± 0.096 / 0.740 ± 0.078 / 0.874 ± 0.041 |
| 0.8 | 0.816 ± 0.021 / 0.823 ± 0.033 / 0.837 ± 0.022 / 0.864 ± 0.031 |
| 1.5 | 0.873 ± 0.021 / 0.874 ± 0.030 / 0.863 ± 0.012 / 0.903 ± 0.009 |

These are small simulator runs, not evidence that the GAE will perform similarly
on real patients. The accuracy values are also affected by the sparse positive
rate, so hidden-cell AUC and recall@K are more informative reconstruction
comparisons. The GAE improves over prevalence ranking as simulated patient
structure increases, but its downstream LSTM AUC does not beat both fill
baselines at every strength. The fully-observed condition is a reference, not a
deployable missing-data method.

The NumPy-only imputation workflow runs without TensorFlow. The downstream
repo-model comparison uses TensorFlow and the actual `main.py` entry point.
This repository was written against TensorFlow 1.x APIs (`tf.app.flags`,
sessions, `tf.nn.rnn_cell`, and `tf.contrib`) which are absent from modern
TensorFlow. `legacy_tf.py` supplies the small TF1 compatibility surface used by
these models, and `requirements-repo-models-legacy.txt` pins a tested
macOS-arm64 TensorFlow build (`tensorflow-macos==2.12.0`) plus scikit-learn.
Install that file to run the four actual LSTM training/evaluation jobs:

```sh
python3 -m pip install -r requirements-repo-models-legacy.txt
python3 graph_imputation.py --patients 600 --missing-rate 0.3 --epochs 80 \
  --model-epochs 20 --prediction-model LSTM
```

TensorFlow model execution is substantially slower than the NumPy imputation
stage. If TensorFlow is missing or the legacy entry point fails, the command
still reports imputation results and sets `pregnancy_prediction.status` to
`unavailable` with the requested model and runtime error; it never substitutes
a different classifier. Use `--skip-downstream` for imputation-only runs.


## Citation

If you use this work, please cite:

Bo Liu, Shuyang Shi, Yongshang Wu, Daniel Thomas, Laura Symul, Emma Pierson and Jure Leskovec. "Predicting pregnancy using large-scale data from a women's health tracking mobile application". WWW 2019.

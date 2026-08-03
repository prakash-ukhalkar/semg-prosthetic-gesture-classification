# Methodology

This document describes the end-to-end methodology implemented in this repository for
sEMG-based prosthetic gesture classification. It reflects the pipeline as actually built
in `src/` and executed in `notebooks/01`–`15` (parameters sourced from `src/config.py`).

---

## 1. Dataset

- **Source:** NinaPro Database 2 (DB2), a public benchmark dataset for sEMG-based
  hand/wrist gesture recognition.
- **Subjects:** 40
- **Gestures:** 50 movement classes (plus rest)
- **Channels:** 12 sEMG channels
- **Sampling rate:** 2000 Hz

Dataset acquisition and integrity verification are handled in **NB01** (project setup and
dataset acquisition) and **NB02** (dataset characterization — file inventory, storage
footprint, structural audit).

## 2. Signal Preprocessing

Implemented in `src/filtering.py`, `src/preprocessing.py`, `src/normalization.py`; executed
in **NB03**.

- **Band-pass filtering:** 4th-order Butterworth, zero-phase (`filtfilt`), 20–500 Hz.
  - 20 Hz lower cutoff removes low-frequency motion artefacts.
  - 500 Hz upper cutoff removes noise outside the physiological sEMG range.
- **Power-line notch filtering:** 50 Hz notch (Q = 30), matching the European recording
  environment of NinaPro DB2.
- **Normalization:** Z-score standardization by default (`NORMALIZATION_METHOD = "zscore"`);
  Min-Max, Robust, and Unit-Vector scaling are also implemented as configurable alternatives.
- Signal quality checks (outlier detection, constant-channel detection) are run prior to
  filtering to flag corrupted recordings.

## 3. Windowing and Segmentation

Implemented in `src/segmentation.py`; executed in **NB04**.

- **Window size:** 200 ms (balances classification latency against feature representation
  quality — standard in the sEMG gesture-recognition literature).
- **Window stride/overlap:** 50 ms increment, giving continuous, high-density predictions.
- **Label propagation:** majority-vote policy assigns each window the most frequent ground
  truth label among its constituent samples (`center` and `strict` policies also supported).
- **Boundary handling:** incomplete trailing windows are dropped (`SEGMENTATION_PADDING = "drop"`).

## 4. Feature Engineering

Three complementary feature domains are extracted per window per channel
(`src/features/time_domain.py`, `frequency_domain.py`, `wavelet.py`), executed in
**NB05** and **NB06**.

**Time-domain features** (NB05): Mean Absolute Value (MAV), Root Mean Square (RMS),
Waveform Length (WL), Zero Crossings (ZC), Slope Sign Changes (SSC), Integrated EMG (IEMG),
Variance (VAR), Willison Amplitude (WAMP). Zero-crossing/slope-sign-change/amplitude
thresholds are standardized in Z-score units (`ZC_THRESHOLD`, `SSC_THRESHOLD`,
`WAMP_THRESHOLD` = 0.01) to remain noise-robust across channels and subjects.

**Frequency-domain features** (NB06): Mean Frequency (MNF), Median Frequency (MDF), Peak
Frequency (PKF), Mean Power, and sub-band power computed via FFT/PSD across three
physiologically-motivated frequency bands (20–150 Hz, 150–300 Hz, 300–500 Hz), plus a
low/high frequency-ratio feature split at 150 Hz.

**Wavelet features** (NB06): Discrete Wavelet Transform (DWT) using the Daubechies-4
(`db4`) wavelet at decomposition level 4 — a wavelet family widely reported as well-suited
to sEMG's non-stationary, transient signal characteristics.

Feature extraction is vectorized over 3D window tensors and validated against closed-form
reference calculations in `tests/test_time_domain_features.py`,
`test_frequency_features.py`, and `test_wavelet_features.py`.

## 5. Feature Fusion and Selection

Implemented in `src/feature_selection/` (`fusion.py`, `quality.py`, `filters.py`,
`rankers.py`, `pipeline.py`, `metadata.py`); executed in **NB07**.

1. **Fusion:** time-domain and frequency/wavelet feature databases are merged per subject
   into a unified feature matrix.
2. **Quality validation:** checks for NaNs, infinities, constant columns, duplicate
   columns, and outliers.
3. **Multi-stage filtering:**
   - Stage 1 — remove constant features.
   - Stage 2 — remove near-zero-variance features.
   - Stage 3 — remove highly correlated features (Pearson correlation threshold).
4. **Ranking and consensus selection:** Minimum Redundancy Maximum Relevance (mRMR),
   Random Forest importance, and XGBoost importance rankers are combined into a consensus
   ranking. Two reduced feature sets are produced (top-25 and top-50 features per channel)
   and cached as Parquet feature databases for downstream reuse.

## 6. Classical Machine Learning Benchmarking

Implemented in `src/ml/models.py`, `pipelines.py`, `splits.py`, `benchmark.py`,
`training.py`, `metrics.py`; executed in **NB08**.

- **Classifiers benchmarked (15):** Logistic Regression, Linear Discriminant Analysis
  (LDA), Quadratic Discriminant Analysis (QDA), Gaussian Naive Bayes, K-Nearest Neighbors,
  Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost,
  LightGBM, CatBoost, Linear SVM, RBF SVM.
- **Feature sets:** top-25 and top-50 consensus-selected features → 15 × 2 = 30 benchmark
  experiments.
- **Splitting strategy:** subject-disjoint 70/15/15 train/validation/test split via
  `GroupShuffleSplit`, grouped by subject ID, to prevent identity leakage between splits.
- Scaling is integrated into scikit-learn `Pipeline` objects (fit only on training data)
  to further prevent leakage. Benchmark runs are resumable via on-disk checkpointing.

No deep learning models are used anywhere in the pipeline — this is an intentional design
choice to test whether tuned classical ML with rich multi-domain features can match
deep-learning-level performance at a fraction of the compute/power cost.

## 7. Hyperparameter Optimization

Implemented in `src/ml/optimization.py`, `search_space.py`, `objectives.py`, `trials.py`,
`early_stopping.py`; executed in **NB09**.

- **Method:** Optuna, Tree-structured Parzen Estimator (TPE) sampler.
- **Models tuned:** the top 3 gradient-boosted decision tree models from the NB08
  benchmark — CatBoost, XGBoost, LightGBM.
- **Objective:** Macro F1-score (chosen over accuracy to fairly weight all 50 gesture
  classes despite class imbalance).
- **Trials:** `OPTUNA_TRIALS = 150` per model.

## 8. Final Model Evaluation

Implemented in `src/ml/evaluation_pipeline.py`; executed in **NB10**.

- Evaluation on a fully held-out subject-disjoint test set (6 unseen subjects).
- Metrics: Macro/weighted F1, precision, recall, ROC/PR curves, confusion matrices,
  bootstrapped confidence intervals.
- Calibration assessment: Expected Calibration Error (ECE) and Brier score.
- Statistical comparison between models via McNemar's test.
- Per-class and physiologically-grounded error analysis (which gestures/muscle groups are
  hardest to classify).

## 9. Cross-Subject Generalization (LOSO)

Implemented in `src/ml/loso.py`, `cross_subject.py`, `subject_analysis.py`,
`fold_metrics.py`; executed in **NB11**.

- **Protocol:** Leave-One-Subject-Out cross-validation — 40 folds, one per subject, applied
  to the Optuna-tuned CatBoost model across all 50 gesture classes.
- Fold-level checkpointing supports resumable execution over the full 692,276-window
  dataset.
- Reports per-fold and aggregate stability statistics, subject ranking, and an aggregate
  confusion matrix, explicitly separating **intra-subject** performance (train/test on the
  same subject) from **cross-subject transfer** performance (train on 39 subjects, test on
  the held-out subject) — the latter is markedly harder and is the basis for the project's
  clinical recalibration recommendation.

## 10. Explainable AI (XAI)

Implemented in `src/explainability.py`; executed in **NB12**.

- **Method:** TreeSHAP (SHAP) for both global (dataset-level) and local (single-prediction)
  feature attribution on the final tuned model.
- Permutation importance computed as a cross-check against SHAP rankings.
- Spearman rank-correlation is computed between the NB07 filter/consensus feature ranking
  and the NB12 SHAP-derived ranking, to validate that the classical feature-selection stage
  agrees with a model-agnostic explainability method.
- Channel × feature-family attribution matrices link SHAP importance back to physical sEMG
  electrode locations for clinical interpretability.

## 11. Ablation Studies

Executed in **NB13** (post-hoc analysis, reuses cached artifacts without retraining):

- Feature-removal ablation (impact of dropping individual features/feature families on
  performance).
- Channel-efficiency / hardware-complexity ablation (performance vs. number of active
  sEMG channels), relevant to minimizing electrode count in a real prosthetic socket.

## 12. Deployment and Model Optimization

Executed in **NB14**, targeting low-power embedded/edge deployment:

- **Export formats compared:** native Pickle, GZIP/ZIP compression, CatBoost native binary
  (`.cbm`), C++ header source, and ONNX.
- **Quantization:** post-training INT8/FP16 quantization, with prediction-consistency
  validation between the native Python model and the exported/quantized model.
- **Benchmarking:** single-sample and batch inference latency/throughput, targeting
  microcontroller/embedded-ARM-class hardware rather than GPU-class edge devices.

## 13. Publication Package Assembly

Executed in **NB15**: verifies completeness of all artifacts from NB01–14, builds a master
results database and master figure/table indexes, and exports submission-ready assets in
LaTeX, CSV, and Markdown formats.

---

## Reproducibility

- Global random seed fixed at `RANDOM_STATE = 42` throughout (data splitting, model
  initialization, Optuna sampling).
- All subject-level splits are group-disjoint (no subject appears in more than one of
  train/validation/test, or in more than one LOSO fold as both train and test).
- Environment pinned via `requirements.txt` / `environment.yml` (see `REPRODUCIBILITY.md`
  for full setup instructions).

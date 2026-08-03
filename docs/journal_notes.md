# Journal Notes

Weekly progress, research findings, and project milestones. Entries are appended
chronologically and should not be edited retroactively — corrections go in a new entry.

---

## 2026-07-26 — Repository & Data Scaffolding

- Repository initialized: `README.md`, `LICENSE`, directory scaffolding for `data/`,
  `models/`, `outputs/`.
- Decision: NinaPro Database 2 (DB2) selected as the primary dataset — 40 subjects, 50
  gesture classes, 12 sEMG channels, 2000 Hz sampling rate. Chosen as the most widely used
  public benchmark for sEMG gesture recognition, maximizing comparability with prior work.

## 2026-07-27 — Data Pipeline Foundations (NB01-NB04)

- **NB01 (Project Setup & Dataset Acquisition):** environment/venv/seed/logging
  verification; raw NinaPro DB2 file integrity checks.
- **NB02 (Dataset Characterization):** full audit of DB2 — file inventory, storage
  footprint, directory structure report.
- **NB03 (Signal Preprocessing):** built the filtering pipeline — 4th-order Butterworth
  band-pass (20-500 Hz) + 50 Hz notch filter (Q=30), zero-phase (`filtfilt`); raw signal
  quality assessment (outlier/constant-channel detection).
- **NB04 (Windowing & Segmentation):** sliding-window segmentation implemented — 200 ms
  windows, 50 ms stride, majority-vote label propagation.

## 2026-07-30 — Feature Engineering & Selection (NB05-NB08)

- **NB05 (Time-Domain Features):** MAV, RMS, WL, ZC, SSC, IEMG, VAR, WAMP extracted per
  window per channel.
- **NB06 (Frequency & Wavelet Features):** MNF, MDF, PKF, band power (FFT/PSD-based) across
  three physiological sub-bands, plus DWT (db4, level 4) wavelet features.
- **NB07 (Feature Fusion & Selection):** merged time/frequency/wavelet feature databases
  subject-by-subject; multi-stage selection (constant/near-zero-variance/correlation
  filters → mRMR + Random Forest + XGBoost consensus ranking) producing top-25 and top-50
  feature sets, cached as Parquet.
- **NB08 (ML Benchmark Framework):** benchmarked 15 classical classifiers (Logistic
  Regression, LDA, QDA, Gaussian NB, KNN, Decision Tree, Random Forest, Extra Trees,
  AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Linear SVM, RBF SVM) across both
  feature sets = 30 experiments total, using subject-disjoint 70/15/15 `GroupShuffleSplit`.

## 2026-07-31 — Tuning, Evaluation, Generalization, and Explainability (NB09-NB12)

- **NB09 (Hyperparameter Optimization):** Optuna TPE search (150 trials) on the top 3
  gradient-boosted tree models from NB08 — CatBoost, XGBoost, LightGBM — optimizing Macro
  F1 to account for 50-class imbalance.
- **NB10 (Final Model Evaluation):** held-out subject-disjoint test evaluation (6 unseen
  subjects, ~103,709 windows). CatBoost identified as the strongest tuned model. Best
  recognized class: resting state; hardest classes: deep/thin forearm muscle gestures
  (thumb, ring finger) — consistent with known sEMG cross-talk/depth limitations.
- **NB11 (Cross-Subject Generalization / LOSO):** full 40-fold Leave-One-Subject-Out
  cross-validation of the tuned CatBoost model across all 40 subjects / 50 gestures /
  692,276 windows, with fold-level checkpointing for resumable execution. Confirms a large
  gap between intra-subject and cross-subject transfer performance — this becomes a central
  finding for the manuscript's clinical-recalibration argument.
- **NB12 (Explainable AI):** TreeSHAP global/local explanations and permutation importance
  on the final model; Spearman rank-correlation between the NB07 filter/consensus feature
  ranking and the NB12 SHAP ranking, plus a channel × feature-family attribution matrix for
  physiological interpretability.

## Status as of 2026-07-31

- NB01-NB12 are implemented and committed to git.
- NB13 (Ablation Studies), NB14 (Deployment & Model Optimization), and NB15 (Publication
  Package) are implemented in the working tree but **not yet committed**.
- Dependency files (`requirements.txt`, `environment.yml`, `pyproject.toml`) updated to pin
  the actual installed versions (CatBoost, LightGBM, Optuna, PyWavelets, PyArrow, psutil,
  etc. were previously used in the pipeline but missing from the pinned dependency lists —
  now corrected).
- `semg-venv/` (local virtual environment) added to `.gitignore`.
- Next actions: commit NB13-15; populate `docs/LITERATURE_MATRIX.md` and
  `docs/references.bib` with a real literature review pass; move manuscript drafting
  forward in `publication/`.

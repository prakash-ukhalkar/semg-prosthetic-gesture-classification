# REPRODUCIBILITY.md

# Reproducibility Protocol & Execution Guide

**Project Title:** Machine Learning-Based sEMG Prosthetic Gesture Classification Using the Ninapro DB5 Dataset  
**Target Journal:** *Series on Biomechanics* (and specialized Biomedical Engineering / Assistive Technology Journals)  
**Repository Version:** `<VERSION>`  
**Release Date:** `<RELEASE_DATE>`  

---

## 1. Project Overview

This repository provides an end-to-end, reproducible research pipeline for surface electromyography (sEMG) signal processing, feature engineering, machine learning gesture classification, Leave-One-Subject-Out (LOSO) cross-subject evaluation, Explainable AI (XAI) analysis, and model optimization for prosthetic control applications.

The entire pipeline is engineered following the **FAIR** (Findable, Accessible, Interoperable, and Reusable) principles for scientific data and software. All algorithmic procedures—from raw sEMG signal filtering to high-dimensional feature fusion and model deployment—are encapsulated in a modular Python source package (`src/`) and demonstrated sequentially across 15 Jupyter Notebooks (`notebooks/`).

---

## 2. Research Objectives

The research methodology implemented in this repository addresses five core biomechanical and computational questions:
1. **Classifier Benchmarking:** Evaluating classical and ensemble machine learning algorithms (SVM, Random Forest, XGBoost, LightGBM, CatBoost, KNN, MLP) for sEMG gesture recognition.
2. **Feature Engineering & Selection:** Identifying the most discriminative time-domain, frequency-domain, and time-frequency (wavelet) sEMG feature combinations.
3. **Cross-Subject Generalization:** Quantifying model performance decay and domain shift across subjects using Leave-One-Subject-Out (LOSO) cross-validation.
4. **Model Explainability (XAI):** Utilizing SHAP (SHapley Additive exPlanations) to interpret biomechanical channel contributions and feature importance.
5. **Deployment Optimization:** Assessing trade-offs between model accuracy, execution latency, memory footprint, and ONNX quantization for real-time prosthetic control.

---

## 3. Repository Structure

```text
semg-prosthetic-gesture-classification/
├── REPRODUCIBILITY.md                # Detailed execution & reproducibility protocol
├── CITATION.cff                      # Machine-readable software citation file
├── CODE_OF_CONDUCT.md                # Contributor covenant code of conduct
├── CONTRIBUTING.md                   # Developer & research contribution guidelines
├── PROJECT_REFERENCE.md              # Project roadmap & structural specifications
├── README.md                         # Main repository overview & quick start
├── LICENSE                           # Open-source MIT License
├── pyproject.toml                    # Build system & package metadata
├── requirements.txt                  # PIP package dependencies
├── environment.yml                   # Conda environment specification
├── .gitignore                        # Git exclusion parameters
│
├── src/                              # Core Python source library
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Global parameters, sampling rates, paths
│   ├── dataset.py                    # Ninapro DB5 data downloaders and parsers
│   ├── preprocessing.py              # Baseline drift correction & artifact removal
│   ├── filtering.py                  # Bandpass & notch filtering implementations
│   ├── normalization.py              # Z-score, Min-Max, and MVC normalization
│   ├── segmentation.py               # Sliding window signal segmentation
│   ├── explainability.py             # SHAP & feature attribution routines
│   ├── utils.py                      # Logging, file I/O, seed setters
│   │
│   ├── features/                     # Feature extraction modules
│   │   ├── time_domain.py            # MAV, RMS, WL, ZC, SSC, IEMG, VAR, WAMP
│   │   ├── frequency_domain.py       # MNF, MDF, PKF, MNP, VDF, Spectral Energy
│   │   └── wavelet.py                # Discrete Wavelet Transform (DWT)
│   │
│   ├── feature_selection/            # Feature fusion & quality pipeline
│   │   ├── filters.py                # Mutual Information, ANOVA, Variance filters
│   │   ├── rankers.py                # Random Forest / XGBoost importance, mRMR
│   │   ├── fusion.py                 # Multi-domain feature fusion
│   │   ├── quality.py                # Feature quality checks & redundancy removal
│   │   └── pipeline.py               # Automated feature selection pipeline
│   │
│   └── ml/                           # ML framework & benchmarking engine
│       ├── models.py                 # Classifier definitions & wrappers
│       ├── benchmark.py              # Model benchmarking suite
│       ├── optimization.py           # Optuna hyperparameter tuning routines
│       ├── evaluation_pipeline.py    # Cross-validation & test evaluation loops
│       ├── loso.py                   # Leave-One-Subject-Out (LOSO) engine
│       ├── cross_subject.py          # Cross-subject generalization evaluation
│       └── visualization.py          # Confusion matrices, ROC curves, decision boundaries
│
├── notebooks/                        # Sequential research notebooks (01 to 15)
│   ├── 01_project_setup_and_dataset_acquisition.ipynb
│   ├── 02_dataset_characterization.ipynb
│   ├── 03_signal_preprocessing.ipynb
│   ├── 04_windowing_and_segmentation.ipynb
│   ├── 05_time_domain_feature_engineering.ipynb
│   ├── 06_frequency_and_time_frequency_feature_engineering.ipynb
│   ├── 07_feature_fusion_and_selection.ipynb
│   ├── 08_ml_benchmark_framework.ipynb
│   ├── 09_hyperparameter_optimization.ipynb
│   ├── 10_final_model_evaluation.ipynb
│   ├── 11_cross_subject_generalization_LOSO.ipynb
│   ├── 12_explainable_ai.ipynb
│   ├── 13_ablation_studies.ipynb
│   ├── 14_deployment_and_model_optimization.ipynb
│   └── 15_publication_package.ipynb
│
├── docs/                             # Engineering & literature documentation
├── tests/                            # Automated unit & integration tests (`pytest`)
├── data/                             # Raw, interim, & processed data (git-ignored)
├── models/                           # Trained model checkpoints & scalers (git-ignored)
└── outputs/                          # Generated figures, tables, & reports
```

---

## 4. System & Computing Environment

### 4.1 Hardware Specifications
The pipeline was developed and benchmarked under the following hardware configuration:
* **Processor (CPU):** Intel Core i7 / AMD Ryzen 7 (8+ physical cores recommended for multi-threaded feature extraction and cross-validation)
* **Memory (RAM):** 16 GB minimum (32 GB recommended for full LOSO feature matrix processing)
* **Storage:** 20 GB available SSD space (for raw Ninapro DB5 files, extracted feature tables, and model checkpoints)
* **Graphics Processing Unit (GPU):** Optional (NVIDIA CUDA-compatible GPU with 6+ GB VRAM accelerates XGBoost/CatBoost model training and SHAP calculation, but is not strictly required).

### 4.2 Operating System
The software has been validated across:
* **Windows:** Windows 10 / Windows 11 (64-bit)
* **Linux:** Ubuntu 20.04 LTS / 22.04 LTS (64-bit)
* **macOS:** macOS 12 (Monterey) or later (Apple Silicon / Intel)

### 4.3 Software & Python Dependencies
* **Python Runtime:** Python 3.12+ (Python 3.10+ fully supported)
* **Core Libraries:**
  * Numerical Computing & Data Structures: `numpy>=1.26.0`, `scipy>=1.11.0`, `pandas>=2.1.0`
  * Signal Processing & Wavelets: `PyWavelets>=1.4.1`
  * Machine Learning Frameworks: `scikit-learn>=1.3.0`, `xgboost>=2.0.0`, `lightgbm>=4.1.0`, `catboost>=1.2.0`
  * Hyperparameter Optimization: `optuna>=3.4.0`
  * Explainable AI (XAI): `shap>=0.43.0`
  * Deployment & Runtime: `onnxruntime>=1.16.0`
  * Visualization: `matplotlib>=3.8.0`, `seaborn>=0.13.0`
  * Testing & Quality Assurance: `pytest>=7.4.0`

---

## 5. Environment Setup Instructions

### Option A: Using Conda (Recommended)

```bash
# Clone the repository
git clone <GITHUB_REPOSITORY_URL>.git
cd semg-prosthetic-gesture-classification

# Create conda environment from specification file
conda env create -f environment.yml

# Activate environment
conda activate semg-venv

# Install local package in editable mode
pip install -e .
```

### Option B: Using Standard Python Virtual Environment (`venv`)

```bash
# Clone the repository
git clone <GITHUB_REPOSITORY_URL>.git
cd semg-prosthetic-gesture-classification

# Create virtual environment
python -m venv semg-venv

# Activate virtual environment
# Windows (PowerShell):
.\semg-venv\Scripts\Activate.ps1
# Linux / macOS:
source semg-venv/bin/activate

# Upgrade package management tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install local package in editable mode
pip install -e .
```

---

## 6. Dataset Acquisition & Integrity Verification

### 6.1 Dataset Access & Licensing Policy
This project utilizes the **Ninapro DB5 (Database 5)** dataset. In compliance with data protection policies and original dataset licensing terms:
* **The raw Ninapro dataset is NOT redistributed in this repository.**
* Users must independently download the original dataset files directly from the official Ninapro repository portal:
  * **Official Source:** [Ninapro Repository Portal](https://ninapro.hevs.ch/)
  * **Dataset Identifier:** DB5 (Double Thalmic Myo Armbands, 10 intact subjects, 52 gestures)

### 6.2 Expected File Layout
After downloading Ninapro DB5, place the uncompressed MATLAB (`.mat`) files into the `data/raw/` directory maintaining the following structure:

```text
data/
└── raw/
    ├── S1_E1_A1.mat
    ├── S1_E2_A1.mat
    ├── S1_E3_A1.mat
    ├── S2_E1_A1.mat
    ...
    └── S10_E3_A1.mat
```

### 6.3 Automated Download & Verification Script
Execute the provided dataset initialization script to verify directory integrity:

```bash
python data/download_dataset.py --verify-only
```

---

## 7. Sequential Notebook Execution Order

To reproduce all computational experiments, run the Jupyter notebooks in `notebooks/` sequentially from `01` through `15`. Each notebook is self-contained, documents its methodology, and serializes intermediate states to `data/` or `outputs/`.

| Order | Notebook Name | Input Artifacts | Generated Output Artifacts |
|:---:|:---|:---|:---|
| **01** | `01_project_setup_and_dataset_acquisition.ipynb` | `data/raw/*.mat` | Dataset structure summary & validation logs |
| **02** | `02_dataset_characterization.ipynb` | Raw sEMG arrays | Channel distributions & SNR characterization plots |
| **03** | `03_signal_preprocessing.ipynb` | Raw sEMG arrays | Cleaned & filtered sEMG signals (`data/interim/`) |
| **04** | `04_windowing_and_segmentation.ipynb` | Cleaned signals | Windowed signal tensors ($200\text{ ms}$ length, $50\text{ ms}$ stride) |
| **05** | `05_time_domain_feature_engineering.ipynb` | Windowed signals | Time-domain feature matrix (`data/features/td.parquet`) |
| **06** | `06_frequency_and_time_frequency_feature_engineering.ipynb` | Windowed signals | Spectral & DWT feature matrices (`data/features/tf.parquet`) |
| **07** | `07_feature_fusion_and_selection.ipynb` | Raw feature matrices | Fused & selected feature subset (`data/features/selected.parquet`) |
| **08** | `08_ml_benchmark_framework.ipynb` | Selected feature subset | Baseline model comparison table (`outputs/tables/`) |
| **09** | `09_hyperparameter_optimization.ipynb` | Selected feature subset | Optuna trial database (`outputs/optuna_study.db`) |
| **10** | `10_final_model_evaluation.ipynb` | Tuned model configs | Test set confusion matrices & ROC curves |
| **11** | `11_cross_subject_generalization_LOSO.ipynb` | Full dataset | LOSO cross-subject evaluation logs & boxplots |
| **12** | `12_explainable_ai.ipynb` | Best model checkpoint | SHAP summary plots & feature attribution maps |
| **13** | `13_ablation_studies.ipynb` | Selected feature subset | Feature domain & window length ablation tables |
| **14** | `14_deployment_and_model_optimization.ipynb` | Best model checkpoint | ONNX optimized model files & latency benchmarks |
| **15** | `15_publication_package.ipynb` | All output tables/plots | Publication-ready figures (`outputs/figures/`) & LaTeX tables |

---

## 8. Determinism & Random Seed Policy

To ensure bitwise or statistical reproducibility across different hardware environments:
1. **Global Seed Initialization:** All random number generators in Python (`random`), NumPy (`np.random`), and PyTorch/Scikit-Learn are initialized with a fixed global seed:
   ```python
   SEED = 42
   ```
2. **Machine Learning Classifiers:** Explicit `random_state=SEED` flags are set across all estimators (RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier).
3. **Cross-Validation Splitters:** Stratified K-Fold and subject splits enforce deterministic seed initialization (`n_splits=5, shuffle=True, random_state=42`).

---

## 9. Computational Footprint & Resource Profiling

| Stage / Component | CPU Core Target | Peak RAM | GPU Required? | Estimated Execution Time |
|:---|:---:|:---:|:---:|:---:|
| **Signal Preprocessing (Notebooks 01–04)** | 4 Cores | ~4 GB | No | 3 – 5 minutes |
| **Feature Extraction (Notebooks 05–06)** | 8 Cores (Parallel) | ~8 GB | No | 8 – 15 minutes |
| **Feature Selection (Notebook 07)** | 4 Cores | ~6 GB | No | 4 – 6 minutes |
| **Model Benchmarking (Notebook 08)** | 8 Cores | ~8 GB | Optional | 10 – 20 minutes |
| **Optuna Optimization (Notebook 09)** | 8 Cores | ~12 GB | Recommended | 30 – 60 minutes |
| **LOSO Cross-Validation (Notebook 11)** | 8 Cores | ~16 GB | Recommended | 45 – 90 minutes |
| **SHAP Explainability (Notebook 12)** | 4 Cores | ~8 GB | Optional | 10 – 15 minutes |
| **Full Pipeline Run (Notebooks 01–15)** | **8 Cores** | **~16 GB** | **Optional** | **~2.5 – 3.5 Hours** |

---

## 10. Reproducing Key Study Results

### 10.1 Regenerating Publication Figures & Tables
To execute the automated publication export pipeline without re-running long training loops:

```bash
jupyter nbconvert --to notebook --execute notebooks/15_publication_package.ipynb
```

Generated outputs will populate `outputs/figures/` (high-resolution 300 DPI EPS/PNG files) and `outputs/tables/` (formatted LaTeX `.tex` tables).

### 10.2 Automated Unit & Integration Testing
To verify code logic and mathematical consistency before executing notebooks:

```bash
pytest tests/ -v
```

---

## 11. Known Sources of Non-Determinism & Variability

While every effort has been made to enforce strict determinism, minor floating-point variations may occur due to:
* **Multi-threaded Floating-Point Accumulation:** Parallel reduction order in OpenMP / C++ backends (e.g., XGBoost, LightGBM, CatBoost) across different CPU architectures (AVX-512 vs ARM NEON).
* **BLAS/LAPACK Libraries:** Differences between Intel MKL, OpenBLAS, or Apple Accelerate frameworks across OS distributions.
* **GPU CUDA Non-Determinism:** Atomic floating-point operations in CUDA kernels during parallel tree building or SHAP calculations.

---

## 12. Troubleshooting & FAQ

**Q1: The raw `.mat` files fail to load in Notebook 01.**  
*Solution:* Ensure `scipy>=1.11.0` is installed and that the Ninapro DB5 `.mat` files were not corrupted during download. Verify file paths in `src/config.py`.

**Q2: Out of Memory (OOM) error during Feature Extraction (Notebook 06).**  
*Solution:* Reduce parallel job allocation in `src/config.py` (`N_JOBS=4` or `N_JOBS=2`), or increase system swap space.

**Q3: Optuna study fails to initialize.**  
*Solution:* Ensure SQLite storage file `outputs/optuna_study.db` has write permissions or delete stale database instances before re-running Notebook 09.

---

## 13. Version History & Roadmap

* **v1.0.0 (`<RELEASE_DATE>`):** Initial publication release supporting Ninapro DB5 gesture classification, 15 sequential notebooks, LOSO evaluation, SHAP explainability, and ONNX deployment routines.
* **Future Improvements:**
  * Integration of additional public sEMG datasets (Ninapro DB2, CapgMyo, CSL-HDEMG).
  * Real-time streaming evaluation loop for physical prosthetic hardware prototypes.

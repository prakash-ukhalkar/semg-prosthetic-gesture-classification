# Machine Learning-Based sEMG Prosthetic Gesture Classification

## Overview

This repository implements a complete end-to-end machine learning pipeline for recognizing prosthetic hand gestures using surface electromyography (sEMG) signals.

The project emphasizes reproducible research, modular software engineering, and publication-quality experimentation using publicly available datasets.

---

# Motivation

Surface electromyography (sEMG) is widely used for:

- Intelligent prosthetic control
- Rehabilitation engineering
- Human-computer interaction
- Wearable healthcare
- Exoskeleton control
- Robotics

This project investigates whether classical machine learning algorithms can accurately classify hand gestures using extracted EMG features.

---

# Objectives

- Load public sEMG datasets
- Visualize muscle activation signals
- Preprocess EMG recordings
- Perform signal segmentation
- Extract informative features
- Select optimal feature subsets
- Train multiple ML classifiers
- Compare model performance
- Explain predictions using SHAP
- Generate publication-ready results

---

# Dataset

Primary Dataset:

NinaPro Database

Future versions may include:

- CapgMyo
- BioPatRec
- CSL-HDEMG

---

# Repository Structure

```text
semg-prosthetic-gesture-classification/
├── PROJECT_REFERENCE.md              # Central project roadmap & guidelines
├── README.md                         # Project overview and setup instructions
├── LICENSE                           # Open-source MIT license
├── pyproject.toml                    # Build system & package metadata
├── requirements.txt                  # Python dependency specifications
├── environment.yml                   # Conda environment configuration
├── .gitignore                        # Git exclusion configuration
│
├── src/                              # Core modular Python library
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Global parameters, paths, and sample rates
│   ├── dataset.py                    # NinaPro dataset downloaders and readers
│   ├── preprocessing.py              # Signal cleaning & baseline drift removal
│   ├── filtering.py                  # Bandpass & notch filter implementations
│   ├── normalization.py              # Z-score, Min-Max, and MVC normalization
│   ├── segmentation.py               # Sliding window segmentation routines
│   ├── explainability.py             # SHAP & feature attribution methods
│   ├── utils.py                      # Utilities, logging, and seed setters
│   │
│   ├── features/                     # Feature extraction modules
│   │   ├── time_domain.py            # MAV, RMS, WL, ZC, SSC, IEMG, VAR, WAMP
│   │   ├── frequency_domain.py       # MNF, MDF, PKF, MNP, VDF, Spectral Energy
│   │   └── wavelet.py                # DWT & Wavelet Packet decomposition
│   │
│   ├── feature_selection/            # Feature selection & quality control
│   │   ├── filters.py                # Mutual Information, ANOVA, Variance filters
│   │   ├── rankers.py                # Random Forest / XGBoost importance, mRMR
│   │   ├── fusion.py                 # Multi-domain feature fusion
│   │   ├── quality.py                # Feature quality checks & redundancy removal
│   │   ├── pipeline.py               # End-to-end feature selection pipeline
│   │   └── metadata.py               # Feature metadata & dictionary mappings
│   │
│   └── ml/                           # Machine Learning framework & benchmarking
│       ├── models.py                 # Classifier instantiations (SVM, RF, XGB, LGBM, CatBoost)
│       ├── benchmark.py              # Model benchmarking across feature sets
│       ├── optimization.py           # Optuna hyperparameter tuning workflows
│       ├── evaluation_pipeline.py    # Cross-validation & test evaluation loops
│       ├── loso.py                   # Leave-One-Subject-Out (LOSO) CV engine
│       ├── cross_subject.py          # Cross-subject generalization evaluation
│       ├── subject_analysis.py       # Per-subject variability & accuracy analysis
│       ├── splits.py                 # Subject-wise & stratified data splitters
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
│   ├── 08b_deep_learning_baseline_Colab.ipynb  # DL (1D-CNN) baseline, added after initial numbering was set
│   ├── 09_hyperparameter_optimization.ipynb
│   ├── 10_final_model_evaluation.ipynb
│   ├── 11_cross_subject_generalization_LOSO.ipynb
│   ├── 12_explainable_ai.ipynb
│   ├── 13_ablation_studies.ipynb
│   ├── 14_deployment_and_model_optimization.ipynb
│   └── 15_publication_package.ipynb
│
├── docs/                             # Project & literature documentation
│   ├── DEVELOPMENT_GUIDELINES.md     # Code style & engineering conventions
│   ├── LITERATURE_MATRIX.md          # Comprehensive literature survey matrix
│   ├── NOTEBOOK_DEVELOPMENT.md       # Standards for Jupyter notebook creation
│   ├── PUBLICATION_GUIDELINES.md     # Journal submission guidelines & checklists
│   ├── RESEARCH_LOG.md               # Chronological log of research milestones
│   └── references.bib                # BibTeX references for manuscript
│
├── tests/                            # Automated test suite (`pytest`)
│   ├── test_preprocessing.py         # Signal filtering & cleaning unit tests
│   ├── test_time_domain_features.py  # Time-domain math unit tests
│   ├── test_frequency_features.py    # FFT / Spectral feature tests
│   ├── test_wavelet_features.py      # Wavelet decomposition tests
│   ├── test_feature_selection.py     # Feature selection pipeline tests
│   └── test_ml_framework.py          # Model training & evaluation tests
│
├── data/                             # Data directory (raw, processed, features)
├── models/                           # Model checkpoints & fit scalers
└── outputs/                          # High-res figures, tables & publication assets
```

---

# Notebook Workflow

| Notebook | Title & Research Stage | Description |
|-----------|------------------------|-------------|
| **01** | Dataset Setup | Dataset acquisition, NinaPro integration, workspace validation |
| **02** | Characterization | Exploratory signal analysis, channel distributions, subject profiles |
| **03** | Signal Preprocessing | Filtering (bandpass, notch), artifact removal, baseline correction |
| **04** | Windowing & Segmentation | Sliding-window segmentation ($200\text{ ms}$ length, $50\text{ ms}$ stride) |
| **05** | Time-Domain Features | MAV, RMS, WL, ZC, SSC, IEMG, VAR feature extraction |
| **06** | Frequency & Time-Freq Features | FFT spectra, MNF, MDF, PKF, and Discrete Wavelet Transform (DWT) |
| **07** | Feature Fusion & Selection | Multi-domain fusion, redundancy checks, mRMR & feature selection |
| **08** | ML Benchmark Framework | Baseline evaluation of SVM, RF, XGBoost, LightGBM, CatBoost, KNN |
| **09** | Hyperparameter Optimization | Automated Optuna search over classifier hyperparameter spaces |
| **10** | Final Model Evaluation | Test-set performance, confusion matrices, ROC/PR curves |
| **11** | Cross-Subject LOSO | Leave-One-Subject-Out cross-validation for inter-subject transfer |
| **12** | Explainable AI (XAI) | SHAP value calculations, global & local gesture feature attributions |
| **13** | Ablation Studies | Feature-domain ablation, window-size impact, channel reduction |
| **14** | Deployment Optimization | Model quantization, ONNX conversion, inference latency benchmarking |
| **15** | Publication Package | Automated export of publication-ready figures and LaTeX tables |

---

# Machine Learning Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- XGBoost
- LightGBM

---

# Signal Processing

- Filtering
- Normalization
- Windowing
- Feature Extraction
- Feature Selection

---

# Explainable AI

- SHAP
- Feature Importance
- Permutation Importance

---

# Development Environment

Python 3.12+

Virtual Environment

```

python -m venv semg-venv

```

Windows

```

semg-venv\Scripts\activate

```

Linux / macOS

```

source semg-venv/bin/activate

```

---

# Install Dependencies

```

pip install -r requirements.txt

```

---

# Running the Project

Execute notebooks sequentially.

Each notebook saves outputs for the next stage.

---

# Project Workflow

Dataset

↓

Exploration

↓

Cleaning

↓

Filtering

↓

Segmentation

↓

Feature Extraction

↓

Feature Selection

↓

Model Training

↓

Hyperparameter Tuning

↓

Explainable AI

↓

Final Model

↓

Publication Figures

---

# Expected Outputs

- Trained models
- Publication figures
- Performance metrics
- Feature importance
- Saved scalers
- Processed datasets

---

# Code Standards

- PEP8
- Type hints
- Modular functions
- Reusable code
- Well documented

---

# Citation

If this repository contributes to your research, please cite the associated publication once available.

---

# License

MIT License

---

# Future Enhancements

- CNN
- LSTM
- CNN-LSTM
- Transformer
- TinyML
- Real-time prosthetic control
- Transfer Learning
- Cross-subject evaluation
- Multi-dataset benchmarking

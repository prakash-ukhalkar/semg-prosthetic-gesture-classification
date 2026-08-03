# Publication Guidelines

Project:
Machine Learning-Based sEMG Prosthetic Gesture Classification Using Publicly Available Datasets

Version:
1.0

Last Updated:
July 2026

---

# Purpose

This document establishes publication standards for the project.

The objective is to ensure that every notebook, experiment, figure, table, and result generated during development can be directly incorporated into a peer-reviewed journal manuscript with minimal additional work.

This project follows the principles of:

- Reproducible Research
- Open Science
- Transparent Machine Learning
- Publication-Quality Scientific Computing

---

# Target Journals

Primary Target

Series on Biomechanics

Potential Future Journals

Biomedical Signal Processing and Control

Medical & Biological Engineering & Computing

Sensors

Healthcare

Biomedical Engineering Online

Journal of NeuroEngineering and Rehabilitation

IEEE Access

Scientific Reports

---

# Intended Paper Type

Original Research Article

---

# Proposed Working Title

Machine Learning-Based Classification of Prosthetic Hand Gestures Using Surface Electromyography Signals

Alternative Titles

Comparative Analysis of Machine Learning Algorithms for sEMG-Based Prosthetic Gesture Recognition

Feature Engineering and Machine Learning for Surface Electromyography Gesture Classification

Interpretable Machine Learning for Prosthetic Gesture Classification Using Public sEMG Datasets

---

# Target Contribution

The manuscript should contribute more than a simple algorithm comparison.

Possible scientific contributions include

• Comprehensive feature engineering

• Comparative evaluation of classical machine learning algorithms

• Explainable AI analysis

• Feature importance analysis

• Reproducible research workflow

• Modular open-source implementation

---

# Research Questions

RQ1

Can handcrafted sEMG features accurately classify prosthetic hand gestures?

RQ2

Which machine learning classifier performs best?

RQ3

Which features contribute most toward classification?

RQ4

Can Explainable AI improve interpretation of classifier decisions?

RQ5

Can classical machine learning provide competitive performance without deep learning?

---

# Hypotheses

H1

Handcrafted sEMG features provide sufficient discriminative information for accurate gesture classification.

H2

Ensemble classifiers outperform single learners.

H3

Feature selection improves classification performance.

H4

Explainable AI identifies physiologically meaningful features.

---

# Manuscript Structure

1. Title

2. Abstract

3. Keywords

4. Introduction

5. Literature Review

6. Materials and Methods

7. Experimental Setup

8. Results

9. Discussion

10. Conclusion

11. Future Work

12. References

---

# Mapping Between Project and Paper

Notebook 01

↓

Materials

Notebook 02

↓

Dataset Description

Notebook 03

↓

Preprocessing

Notebook 04

↓

Signal Visualization

Notebook 05

↓

Segmentation Methodology

Notebook 06

↓

Time-Domain Features

Notebook 07

↓

Frequency and Wavelet Features

Notebook 08

↓

Feature Selection

Notebook 09

↓

Baseline Experiments

Notebook 10

↓

Advanced Models

Notebook 11

↓

Hyperparameter Optimization

Notebook 12

↓

Explainability

Notebook 13

↓

Results

Notebook 14

↓

Final Pipeline

Notebook 15

↓

Figures and Tables

---

# Experimental Design

Every experiment must specify

Dataset

Subject IDs

Gesture Classes

Sampling Frequency

Window Size

Overlap

Normalization Method

Filtering Method

Feature Set

Classifier

Random Seed

Cross Validation Strategy

Evaluation Metrics

---

# Dataset Reporting

Always report

Dataset name

Version

Number of subjects

Number of gestures

Number of gestures

Number of channels

Sampling rate

Recording protocol

Train/Test split

Class distribution

---

# Statistical Reporting

Whenever classifiers are compared

Report

Mean

Standard Deviation

95% Confidence Interval (if applicable)

Use appropriate statistical tests where justified

Examples

Wilcoxon Signed-Rank Test

Paired t-test

Friedman Test

McNemar Test

Document why the chosen test is appropriate.

---

# Evaluation Metrics

Always report

Accuracy

Precision

Recall

F1-score

Specificity (if relevant)

ROC-AUC (when applicable)

Confusion Matrix

Training Time

Prediction Time

Memory Usage (if measured)

---

# Cross Validation

Preferred

Stratified K-Fold

Repeated Stratified K-Fold

Leave-One-Subject-Out (recommended for subject-independent evaluation)

Avoid reporting results from a single train-test split unless justified.

---

# Figure Standards

All figures should be publication quality.

Minimum Resolution

300 DPI

Preferred

600 DPI

Format

PNG

PDF (vector) when possible

SVG for line graphics

Every figure must include

Figure Number

Title

Axis Labels

Units

Legend

Readable Font Size

Consistent Formatting

---

# Figure Naming Convention

figure_01_dataset_distribution.png

figure_02_signal_example.png

figure_03_preprocessing.png

figure_04_windowing.png

figure_05_feature_importance.png

figure_06_confusion_matrix.png

figure_07_roc_curve.png

---

# Table Standards

Every table should include

Table Number

Title

Units

Proper Formatting

Consistent Decimal Places

Save tables as

CSV

Excel

LaTeX (optional)

---

# Recommended Tables

Dataset Summary

Feature Summary

Hyperparameter Settings

Classifier Comparison

Cross Validation Results

Feature Importance Rankings

Computation Time

Statistical Test Results

---

# Reproducibility Checklist

Every experiment must record

Random Seed

Library Versions

Operating System

Python Version

Dataset Version

Preprocessing Parameters

Feature Parameters

Classifier Parameters

Evaluation Protocol

---

# Code Availability

The repository should include

README

Requirements

Project Reference

Development Guidelines

Publication Guidelines

Notebook Workflow

Source Code

Saved Models

Outputs

License

---

# Explainability

Preferred methods

SHAP

Permutation Importance

Feature Importance

Interpret results in terms of muscle activation and biomechanics whenever possible.

---

# Discussion Guidelines

Interpret results scientifically.

Do not simply report accuracy.

Discuss

Strengths

Limitations

Clinical relevance

Computational cost

Generalizability

Comparison with existing literature

---

# Future Work

Potential extensions

Deep Learning

CNN

LSTM

CNN-LSTM

Transformers

TinyML

Real-Time Prosthetic Control

Embedded Systems

Transfer Learning

Cross-Dataset Evaluation

Domain Adaptation

Multi-Modal Learning

---

# Ethical Considerations

Only publicly available datasets will be used.

No new human participant data will be collected.

Proper citation of all datasets is mandatory.

Respect dataset licenses.

---

# Citation Policy

Cite

Original dataset publication

Feature extraction methods

Machine learning algorithms where appropriate

Software libraries when required

Related literature

---

# Publication Checklist

Before writing the manuscript, verify that:

✓ All notebooks execute successfully from start to finish.

✓ All figures are publication quality.

✓ All tables are complete and labeled.

✓ Every experiment is reproducible.

✓ Random seeds are fixed.

✓ Models are saved.

✓ Hyperparameters are documented.

✓ Evaluation metrics are complete.

✓ Results are interpreted.

✓ Limitations are discussed.

✓ References are verified.

✓ Code is documented.

✓ Repository is publicly shareable.

---

# Writing Style

Use

Clear

Concise

Objective

Scientific

Avoid

Marketing language

Unsubstantiated claims

Overstatement

Unsupported conclusions

---

# Final Goal

The completed repository should enable another researcher to:

1. Download the repository.

2. Install dependencies.

3. Execute every notebook sequentially.

4. Reproduce every experiment.

5. Generate all figures and tables.

6. Verify all reported results.

7. Extend the work for future research.

The repository should serve as both a reproducible research artifact and the companion codebase for the associated journal publication.

End of Document

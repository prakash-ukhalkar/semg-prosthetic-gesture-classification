# Development Guidelines

Project:
Machine Learning-Based sEMG Prosthetic Gesture Classification Using Publicly Available Datasets

Version:
1.0

---

# Purpose

This document defines the software engineering, research, and documentation standards that must be followed throughout this project.

The objective is to ensure that the project remains:

- Reproducible
- Modular
- Maintainable
- Scientifically rigorous
- Publication-ready
- Easy to understand
- Easy to extend

All contributors, including AI coding assistants, should follow these guidelines.

---

# Development Philosophy

This repository is intended to support:

- Academic research
- Journal publication
- Reproducible experiments
- Open-source software

The repository should resemble a professional university research laboratory project rather than a collection of scripts.

---

# General Principles

Always prefer:

Readability > Cleverness

Reproducibility > Convenience

Modularity > Code Duplication

Scientific Correctness > Shortcuts

Documentation > Assumptions

---

# Repository Structure

Never change the project structure without justification.

Reusable code belongs in:

src/

Experimental work belongs in:

notebooks/

Generated outputs belong in:

outputs/

Saved models belong in:

models/

Documentation belongs in:

docs/

Testing belongs in:

tests/

---

# Python Version

Python >= 3.12

---

# Virtual Environment

Always use

semg-venv

Never install packages globally.

---

# Coding Standards

Follow:

PEP8

Maximum line length:

88 characters

Indentation:

4 spaces

Encoding:

UTF-8

---

# Naming Conventions

## Variables

Good

signal_data

window_length

feature_matrix

Bad

x

abc

temp

---

## Functions

Use snake_case

Examples

load_dataset()

extract_features()

train_model()

evaluate_classifier()

---

## Classes

Use PascalCase

Examples

SignalProcessor

FeatureExtractor

RandomForestTrainer

---

## Constants

UPPER_CASE

Examples

WINDOW_SIZE

SAMPLING_RATE

RANDOM_STATE

---

# Type Hints

Always use type hints whenever practical.

Example

def normalize_signal(signal: np.ndarray) -> np.ndarray:

---

# Docstrings

Every public function must include

Description

Parameters

Returns

Raises (if applicable)

Example

Use Google-style or NumPy-style docstrings consistently.

---

# Comments

Write comments only when they explain:

WHY

instead of

WHAT

Avoid obvious comments.

---

# Notebook Standards

Every notebook must begin with

Title

Research Objective

Background

Expected Inputs

Expected Outputs

Notebook Workflow

References

---

# Notebook Order

01 Environment

02 Dataset Exploration

03 Signal Preprocessing

04 Visualization

05 Windowing

06 Time Features

07 Frequency Features

08 Feature Selection

09 Baseline Models

10 Advanced Models

11 Hyperparameter Tuning

12 Explainable AI

13 Evaluation

14 Final Pipeline

15 Publication Results

Never skip notebook order.

---

# Notebook Cell Structure

Recommended order

Markdown

↓

Imports

↓

Configuration

↓

Functions (minimal)

↓

Execution

↓

Visualization

↓

Results

↓

Summary

---

# Imports

Standard Library

↓

Third-party Packages

↓

Local src Imports

Example

import os

import numpy as np

import pandas as pd

from sklearn.svm import SVC

from src.preprocessing import normalize_signal

---

# Random Seed

Always use

RANDOM_STATE = 42

Apply consistently to

NumPy

Scikit-Learn

XGBoost

LightGBM

Any other ML library

---

# Logging

Avoid print().

Prefer

logging

Levels

INFO

WARNING

ERROR

CRITICAL

Store logs when appropriate.

---

# Error Handling

Always validate

Input files

Directories

Dataset availability

Array dimensions

Missing values

Data types

Raise informative exceptions.

---

# Dataset Handling

Never modify raw data.

Workflow

Raw

↓

Interim

↓

Processed

Maintain original datasets unchanged.

---

# Signal Processing Standards

Document

Sampling frequency

Window size

Window overlap

Filtering method

Normalization

Scaling

Feature extraction parameters

Every processing step should be reproducible.

---

# Feature Engineering

Group features

Time Domain

Frequency Domain

Wavelet Domain

Entropy Features

Statistical Features

Biomechanical Features

Document formulas whenever possible.

---

# Machine Learning Standards

Always compare multiple classifiers.

Minimum models

Logistic Regression

Decision Tree

Random Forest

KNN

SVM

XGBoost

LightGBM

Never publish results from only one classifier.

---

# Hyperparameter Tuning

Preferred methods

GridSearchCV

RandomizedSearchCV

Document

Search space

Cross-validation strategy

Scoring metric

---

# Evaluation Metrics

Always report

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Classification Report

ROC Curve (when applicable)

AUC

Training Time

Prediction Time

---

# Cross Validation

Prefer

Stratified K-Fold

or

Repeated Stratified K-Fold

Avoid single train/test split whenever possible.

---

# Explainable AI

Preferred methods

SHAP

Permutation Importance

Feature Importance

Document important findings.

---

# Visualization Standards

Use

Matplotlib

Plotly

Avoid cluttered figures.

Every figure must include

Title

Axis Labels

Units

Legend

Grid (when appropriate)

High Resolution

Publication quality

---

# Figure Saving

Save all figures to

outputs/figures/

Naming example

figure_01_dataset_distribution.png

---

# Tables

Save tables

outputs/tables/

Preferred formats

CSV

Excel

LaTeX (optional)

---

# Model Saving

Save using

joblib

Directory

models/

Example

random_forest.pkl

scaler.pkl

---

# Experiment Tracking

Every experiment should record

Date

Notebook

Dataset

Features

Classifier

Parameters

Metrics

Store reports

outputs/reports/

---

# Configuration

Avoid hard-coded values.

Use

src/config.py

Store

Paths

Random seed

Window size

Sampling frequency

Directories

Global settings

---

# Reusable Code

If code exceeds approximately 20 lines or is needed in multiple notebooks, move it into the appropriate module under `src/` and import it rather than duplicating it.

---

# Testing

Critical functions should include unit tests.

Examples

Signal preprocessing

Window segmentation

Feature extraction

Evaluation metrics

---

# Documentation

Update documentation whenever

New notebook

New module

New dataset

New algorithm

is added.

---

# Git Commit Style

Examples

feat: implement window segmentation

feat: add random forest classifier

fix: correct feature extraction bug

docs: update README

refactor: move preprocessing to src

test: add preprocessing tests

---

# Branch Strategy

main

development

feature/<feature-name>

Example

feature/window-segmentation

feature/random-forest

feature/shap-analysis

---

# Research Reproducibility

Every experiment must document

Dataset version

Library versions

Random seed

Preprocessing steps

Feature extraction method

Classifier parameters

Evaluation protocol

This information should be sufficient for another researcher to reproduce the results.

---

# AI Assistant Guidelines

When generating code:

- Read PROJECT_REFERENCE.md first.
- Read README.md.
- Read DEVELOPMENT_GUIDELINES.md.
- Reuse existing modules.
- Do not duplicate functionality.
- Explain scientific reasoning where appropriate.
- Prefer modular implementations.
- Maintain consistency across notebooks.

---

# Future Extensions

The architecture should support future additions such as:

- CNN-based classifiers
- LSTM models
- Transformer models
- Real-time inference
- TinyML deployment
- Embedded systems
- Cross-subject learning
- Transfer learning
- Domain adaptation
- Multi-modal sensor fusion

The project structure should not require major refactoring to accommodate these extensions.

---

# Final Principle

Every notebook, module, figure, table, and experiment should be of sufficient quality to be included directly in a peer-reviewed journal manuscript with minimal additional modification.

End of Document

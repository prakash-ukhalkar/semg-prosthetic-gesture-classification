# Research Log

Project Title

Machine Learning-Based sEMG Prosthetic Gesture Classification Using Publicly Available Datasets

---

Version

1.0

Project Start Date

July 2026

Repository

semg-prosthetic-gesture-classification

Principal Investigator

(Your Name)

Development Environment

Python 3.10+ (pinned dependency versions in `requirements.txt` / `environment.yml`)

Virtual Environment

semg-venv (gitignored — see `REPRODUCIBILITY.md` for setup)

Primary Dataset

NinaPro Database

Primary Language

Python

---

# Purpose

This document serves as the official electronic research notebook for the project.

Every research activity, implementation decision, experiment, issue, improvement, and observation must be recorded here.

The objectives are to:

• Maintain complete research traceability

• Support reproducibility

• Record scientific reasoning

• Document implementation history

• Facilitate manuscript preparation

• Track project progress

---

# Logging Rules

Every significant activity should include:

Date

Notebook

Objective

Files Modified

Implementation Summary

Observations

Results

Challenges

Next Actions

Estimated Completion

Never delete previous entries.

Instead, append new entries chronologically.

---

# Project Timeline

| Phase | Description | Status |
|--------|-------------|--------|
| Phase 1 | Project Planning | Completed |
| Phase 2 | Repository Setup | Completed |
| Phase 3 | Dataset Acquisition | Completed (NB01) |
| Phase 4 | Dataset Exploration | Completed (NB02) |
| Phase 5 | Signal Preprocessing | Completed (NB03) |
| Phase 6 | Signal Segmentation | Completed (NB04) |
| Phase 7 | Feature Engineering | Completed (NB05, NB06) |
| Phase 8 | Feature Selection | Completed (NB07) |
| Phase 9 | Machine Learning | Completed (NB08) |
| Phase 10 | Hyperparameter Optimization | Completed (NB09) |
| Phase 11 | Model Evaluation | Completed (NB10) |
| Phase 12 | Cross-Subject Generalization (LOSO) | Completed (NB11) |
| Phase 13 | Explainable AI | Completed (NB12) |
| Phase 14 | Ablation Studies | Completed (NB13, uncommitted) |
| Phase 15 | Deployment & Model Optimization | Completed (NB14, uncommitted) |
| Phase 16 | Publication Package | In Progress (NB15, uncommitted) |
| Phase 17 | Manuscript Preparation | In Progress (see `publication/`) |

---

# Research Decisions

This section records important design decisions.

Format

Decision ID

Date

Decision

Reason

Impact

Status

Example

Decision-001

Date

2026-07-19

Decision

Use NinaPro Dataset as the primary dataset.

Reason

Most widely used benchmark dataset for sEMG gesture recognition.

Impact

Improves reproducibility and enables comparison with published studies.

Status

Accepted

---

# Experiment Log

Use the following template for every experiment.

---

## Experiment ID

EXP-001

Date

Notebook

Objective

Dataset

Feature Set

Classifier

Hyperparameters

Evaluation Method

Files Modified

Outputs Generated

Observations

Results

Challenges

Lessons Learned

Next Actions

Status

---

# Notebook Progress

| Notebook | Description | Status | Last Updated |
|------------|-------------|---------|--------------|
| 01 | Project Setup & Dataset Acquisition | Completed | 2026-07-27 |
| 02 | Dataset Characterization | Completed | 2026-07-27 |
| 03 | Signal Preprocessing | Completed | 2026-07-27 |
| 04 | Windowing & Segmentation | Completed | 2026-07-27 |
| 05 | Time Domain Feature Engineering | Completed | 2026-07-30 |
| 06 | Frequency & Time-Frequency Feature Engineering | Completed | 2026-07-30 |
| 07 | Feature Fusion & Selection | Completed | 2026-07-30 |
| 08 | ML Benchmark Framework | Completed | 2026-07-30 |
| 09 | Hyperparameter Optimization | Completed | 2026-07-31 |
| 10 | Final Model Evaluation | Completed | 2026-07-31 |
| 11 | Cross-Subject Generalization (LOSO) | Completed | 2026-07-31 |
| 12 | Explainable AI | Completed | 2026-07-31 |
| 13 | Ablation Studies | Completed (uncommitted) | |
| 14 | Deployment & Model Optimization | Completed (uncommitted) | |
| 15 | Publication Package | In Progress (uncommitted) | |

---

# Dataset Log

Record all dataset-related information.

Dataset Name

Version

Download Source

Download Date

License

Number of Subjects

Number of Gestures

Sampling Frequency

Number of Channels

Original Size

Processed Size

Checksum (Optional)

Notes

---

# Feature Engineering Log

Whenever a new feature is implemented, record:

Feature Name

Category

Formula

Reason for Inclusion

Expected Benefit

Notebook

Date Added

Reference

Implementation Status

---

# Machine Learning Log

For every classifier record:

Algorithm

Library Version

Hyperparameters

Training Time

Prediction Time

Cross Validation

Performance Metrics

Saved Model Path

Comments

---

# Hyperparameter Search Log

Classifier

Search Method

Search Space

Best Parameters

Cross Validation Score

Notebook

Observations

---

# Explainability Log

Method

SHAP

Permutation Importance

Feature Importance

Notebook

Key Findings

Important Features

Interpretation

Clinical Relevance

---

# Figure Log

Every publication figure should be recorded.

Figure Number

Filename

Notebook

Purpose

Output Directory

Publication Ready

Revision Required

---

# Table Log

Every publication table should be recorded.

Table Number

Filename

Notebook

Purpose

Output Directory

Publication Ready

Revision Required

---

# Literature Notes

For every important paper record:

Authors

Year

Title

Journal

Dataset

Methods

Results

Limitations

Ideas for Current Project

Citation

---

# Bug Log

Bug ID

Date

Description

Affected Notebook

Root Cause

Resolution

Status

Lessons Learned

---

# Refactoring Log

Date

Files Modified

Reason

Benefits

Impact

---

# Performance Tracking

Track performance over time.

| Experiment | Accuracy | Precision | Recall | F1 | Notes |
|------------|-----------|-----------|--------|----|-------|

---

# Risks

Potential project risks.

Examples

Dataset compatibility

Class imbalance

Signal noise

Overfitting

Limited generalization

Version incompatibility

Document mitigation strategies.

---

# Ideas Backlog

Store ideas for future investigation.

Examples

Additional feature extraction

New classifiers

Deep Learning

Transfer Learning

TinyML

Real-Time Deployment

Cross-Dataset Validation

Multi-Modal Sensor Fusion

Do not remove ideas.

Mark them as:

Planned

In Progress

Completed

Deferred

Rejected

---

# Publication Notes

Maintain notes related to manuscript preparation.

Possible Titles

Novel Contributions

Reviewer Comments

Additional Experiments

Future Work

Target Journals

Submission Timeline

---

# Weekly Progress Summary

At the end of each week record:

Completed Work

Pending Work

Issues

Decisions Made

Next Week Plan

Estimated Project Completion

---

# Milestones

| Milestone | Target Date | Status |
|------------|-------------|--------|
| Repository Created | 2026-07-26 | Completed |
| Dataset Downloaded | 2026-07-27 | Completed |
| Dataset Verified | 2026-07-27 | Completed |
| Preprocessing Completed | 2026-07-27 | Completed |
| Feature Extraction Completed | 2026-07-30 | Completed |
| Baseline Models Completed | 2026-07-30 | Completed |
| Advanced Models Completed (Tuned GBDT) | 2026-07-31 | Completed |
| Cross-Subject Generalization (LOSO) Completed | 2026-07-31 | Completed |
| Explainability Completed | 2026-07-31 | Completed |
| Ablation Studies Completed | | Completed (uncommitted) |
| Deployment / Model Optimization Completed | | Completed (uncommitted) |
| Publication Figures Ready | | In Progress |
| Manuscript Draft Completed | | Pending |
| Internal Review | | Pending |
| Journal Submission | | Pending |

---

# Version History

| Version | Date | Changes |
|----------|------|----------|
| 1.0 | 2026-07-19 | Initial Research Log Created |
| 1.1 | 2026-07-31 | Synced Project Timeline, Notebook Progress, and Milestones tables to actual repository state (NB01-12 committed, NB13-15 completed locally/uncommitted); updated environment/venv fields |

---

# Final Project Checklist

Before submission, confirm:

✓ Repository organized

✓ Documentation complete

✓ Dataset documented

✓ Code reviewed

✓ Random seeds fixed

✓ Experiments reproducible

✓ Figures publication quality

✓ Tables finalized

✓ Models archived

✓ Requirements updated

✓ README complete

✓ Manuscript drafted

✓ References verified

✓ Repository ready for public release

✓ Journal submission package complete

---

# Notes

This document is a living record of the research process.

Every significant implementation, experiment, observation, and decision should be documented.

The goal is to ensure that another researcher can understand, reproduce, validate, and extend the work without ambiguity.

End of Document

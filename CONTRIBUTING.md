# CONTRIBUTING.md

# Contributing Guidelines

Thank you for your interest in contributing to the **Machine Learning-Based sEMG Prosthetic Gesture Classification** repository!

We welcome contributions from biomedical engineers, machine learning researchers, computer scientists, open-source developers, and students. This document outlines the technical standards, workflow conventions, and quality expectations for contributing code, notebooks, documentation, or issue reports.

---

## 1. Repository Philosophy & Goals

This project serves as a scientific research codebase accompanying peer-reviewed journal publications in biomechanics and rehabilitation engineering. All contributions must adhere to three foundational pillars:
1. **Modular Architecture:** Core logic, signal processing algorithms, and machine learning models belong in the Python package (`src/`), NOT embedded inside Jupyter Notebook cells.
2. **Reproducibility First:** All experimental workflows, random seeds, hyperparameter optimization trials, and evaluation pipelines must be deterministic and fully reproducible.
3. **Publication Quality:** All generated figures, comparison tables, and code documentation must meet top-tier academic journal standards (*Series on Biomechanics*, *IEEE TBME*, *Biomedical Signal Processing and Control*).

---

## 2. Setting Up the Development Environment

### 2.1 Fork & Clone
1. Fork the official repository on GitHub: `<GITHUB_REPOSITORY_URL>`
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/semg-prosthetic-gesture-classification.git
   cd semg-prosthetic-gesture-classification
   ```

### 2.2 Virtual Environment & Dependencies
Set up an isolated environment using Conda or standard Python `venv`:

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate semg-venv

# Option B: Python venv
python -m venv semg-venv
source semg-venv/bin/activate  # Linux/macOS
# semg-venv\Scripts\activate   # Windows

# Install developer dependencies in editable mode
pip install -e .[dev]
```

---

## 3. Running Automated Tests

Before submitting any code or pull request, ensure all unit and integration tests pass:

```bash
# Run full test suite with pytest
pytest

# Run tests with coverage report
pytest --cov=src tests/

# Run linting and style checks
flake8 src/ tests/
black --check src/ tests/
```

---

## 4. Notebook Development Standards

To maintain clean, readable, and reproducible notebooks in `notebooks/`:
* **Concise Cells:** Notebook cells should serve for execution orchestrations and markdown commentary. Complex multi-line algorithms MUST be refactored into `src/`.
* **Sequential Integrity:** Notebooks MUST execute top-to-bottom without errors (`Kernel -> Restart & Run All`).
* **Clear Cell Outputs:** Clear temporary or large binary outputs before committing unless the cell output is a key plot or summary table.
* **Header Markdown:** Every notebook must start with a `# Notebook Title`, `## Objectives`, and `## Inputs / Outputs` summary block.

---

## 5. Coding Style & Quality Guidelines

### 5.1 Python Formatting & Standards
* **PEP 8 Compliance:** All code in `src/` and `tests/` must strictly conform to PEP 8 standards.
* **Type Hinting:** Mandatory type annotations on all public functions, classes, and method signatures:
  ```python
  def filter_emg_signal(
      signal: np.ndarray,
      fs: float = 2000.0,
      lowcut: float = 20.0,
      highcut: float = 500.0,
      order: int = 4
  ) -> np.ndarray:
  ```
* **Docstrings:** Use **Google Style Docstrings** for all modules, classes, and functions:
  ```python
  """Filters raw sEMG signal using a 4th-order Butterworth bandpass filter.

  Args:
      signal (np.ndarray): Continuous raw sEMG voltage signal [samples x channels].
      fs (float): Sampling frequency in Hz. Defaults to 2000.0.
      lowcut (float): Lower cutoff frequency in Hz. Defaults to 20.0.
      highcut (float): Upper cutoff frequency in Hz. Defaults to 500.0.
      order (int): Butterworth filter order. Defaults to 4.

  Returns:
      np.ndarray: Bandpass-filtered sEMG signal array.
  """
  ```

---

## 6. Git Workflow & Commit Conventions

### 6.1 Branch Naming Strategy
Use descriptive branch names prefixed by task type:
* `feature/add-wavelet-packet-features`
* `bugfix/loso-indexing-error`
* `docs/update-reproducibility-guide`
* `refactor/optimize-window-segmentation`

### 6.2 Commit Message Format
We follow the **Conventional Commits** specification:
```text
<type>(<scope>): <short summary>

[optional body]
```

**Allowed Types:**
* `feat`: A new feature or algorithm (e.g., `feat(features): add Cepstral Coefficients module`)
* `fix`: A bug fix (e.g., `fix(ml): resolve StratifiedKFold data leakage`)
* `docs`: Documentation updates (e.g., `docs(readme): update setup instructions`)
* `test`: Adding or modifying unit tests (e.g., `test(preprocessing): add test for notch filter attenuation`)
* `refactor`: Code change that neither fixes a bug nor adds a feature (e.g., `refactor(segmentation): vectorise sliding window logic`)

---

## 7. Issue Reporting & Pull Request (PR) Process

### 7.1 Reporting Issues
Before opening an issue, search the issue tracker to ensure it hasn't already been reported. When creating an issue, provide:
* **Bug Reports:** Detailed steps to reproduce, OS/hardware environment, full traceback, and expected vs actual behavior.
* **Feature Requests:** Scientific or engineering justification, proposed API/module placement, and relevant academic references.

### 7.2 Pull Request Submission Checklist
When submitting a PR:
1. Ensure your branch is rebased on the latest `main` branch.
2. Verify all unit tests pass (`pytest tests/`).
3. Confirm code style passes (`flake8` / `black`).
4. Update or add corresponding unit tests in `tests/` for new features.
5. Update relevant documentation in `docs/` or `README.md`.
6. Ensure `.gitignore` policies are respected (no raw `.mat`, `.parquet`, or binary checkpoints committed).

---

## 8. Performance & Reproducibility Requirements

* **Seed Control:** Any PR introducing randomized operations (model initialization, dataset splits, feature selection) MUST accept an explicit `random_state` parameter defaulted to `42`.
* **Memory Management:** Vectorize array operations using NumPy / SciPy; avoid Python `for` loops over multi-channel sEMG samples.
* **Cross-Platform Safety:** Avoid hardcoded OS file path separators; use `pathlib.Path` or `os.path.join`.

---

## 9. Review Process & Licensing

* **Peer Review:** All PRs require review and approval from at least one core maintainer before merging.
* **Licensing:** By contributing to this repository, you agree that your contributions will be licensed under the project's [MIT License](file:///e:/Bio-Mechanics/semg-prosthetic-gesture-classification/LICENSE).
* **Citation & Attribution:** Substantial contributions to core algorithms or research methodology will be formally acknowledged in the repository's [CITATION.cff](file:///e:/Bio-Mechanics/semg-prosthetic-gesture-classification/CITATION.cff) and accompanying journal manuscripts.

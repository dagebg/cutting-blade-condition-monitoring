# Cutting Blade Condition Monitoring – ML Project
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-CC%20BY--SA%203.0-lightgrey)

This repository contains a refactored version of my individual university project on condition monitoring for an industrial cutting blade.  
The original work was implemented in a single notebook; it is now structured as a modular Python project with reusable components.


## Highlights

- Real industrial condition monitoring use case (shrink‑film packaging line, cutting blade degradation).
- End‑to‑end ML pipeline following the CRISP‑DM process (preprocessing → features → modeling → evaluation).
- Three model families: mode classification, degradation regression, and anomaly detection.


## Dataset

This project uses the **One Year Industrial Component Degradation** dataset by **inIT-OWL**, available on Kaggle:
https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation.

The dataset is licensed under **Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)** by the original authors. Please refer to the Kaggle page for full license details.


> Note: The dataset itself is not included in this repository.  
> Please download it from the original source (e.g. Kaggle) and place it in the expected `data/` folder as described in the notebook.

---

## Project structure

```text
.
├─ data/                     # raw / processed data (not in repo)
├─ main_notebook.ipynb       # main analysis & results notebook
├─ src/
│  ├─ preprocessing.py       # loading, cleaning, basic preprocessing
│  ├─ features.py            # feature engineering functions
│  ├─ models.py              # model training & CV routines
│  ├─ evaluation.py          # reusable evaluation functions
│  └─ helpers.py             # small dataset access utilities 
├─ requirements.txt          # Python dependencies
├─ LICENSE.txt 
└─ README.md
```

## Requirements / Tech stack

- Python 3.10+  
- Core libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, kaggle, notebook, jupyterlab (see `requirements.txt` for exact versions).


## Getting started
### 1. Clone the repository
```bash
git clone https://github.com/dagebg/cutting-blade-condition-monitoring.git
cd cutting-blade-condition-monitoring
```

### 2. Create virtual environment (optional, but recommended)
```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt 
```
### 4. Run the notebook
```bash 
jupyter lab
# or 
jupyter notebook
```
Open `main_notebook.ipynb` and execute the cells from top to bottom.


## Module overview

- `preprocessing.py`  
  Functions for loading raw data, basic cleaning, and preparing intermediate tables.

- `features.py`  
  Functions for generating aggregated and scaled feature tables used by all three models.

- `models.py`  
  Training routines for:
  - Random Forest mode classifier with cross‑validation  
  - Ridge regression degradation model  
  - IsolationForest anomaly detector

- `evaluation.py`  
  Centralised evaluation utilities:
  - Classification metrics and confusion matrices for the mode classifier  
  - Regression metrics and residual analysis for the degradation model  
  - Anomaly detection metrics, specificity, and imbalance analysis

- `helpers.py`  
Dataset access utilities: downloading from Kaggle, loading raw CSV files,
  and loading the pre-built aggregated feature table.
---
## License
The **code** in this repository is licensed under the MIT License – see the [LICENSE](LICENSE.txt) file for details.  
The **dataset** is licensed separately under CC BY-SA 3.0 by its original authors.

# Exploratory Data Analysis and Visualization of Cancer Patients Data

Capstone Project — Assignment II: Data Exploration and Visualization

## Project overview

This project explores the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. The dataset contains measurements computed from digitized images of fine needle aspirate (FNA) breast-mass samples. The target is a diagnostic label:

- `0` — malignant
- `1` — benign

This is a diagnostic research dataset, not a clinical decision-making tool. It does not contain personally identifiable patient records.

## Dataset source

1. Original UCI Machine Learning Repository record: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
2. Python access used in the notebook: [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

The notebook records the dataset shape, feature names, class distribution, missing-value checks, and all analytical assumptions.

## Repository contents

- `cancer_patients_eda.ipynb` — executable analysis notebook with outputs/visualizations
- `cancer_patients_eda.py` — equivalent Python script for reproducible execution
- `report.md` — supporting report with methodology, observations, limitations, and conclusions
- `requirements.txt` — Python dependencies
- `figures/` — generated plots after running the notebook or script

## How to run

```bash
python -m pip install -r requirements.txt
python cancer_patients_eda.py
```

To run the notebook:

```bash
jupyter notebook cancer_patients_eda.ipynb
```

The script and notebook use a fixed random seed (`42`) and create the `figures/` directory automatically.

## Units covered

The work is organized to cover the usual five units in an introductory Data Exploration and Visualization course:

1. Data, Python environment, loading, structure, and data quality
2. Descriptive statistics and univariate exploration
3. Bivariate analysis and relationships
4. Multivariate visualization and preprocessing
5. Interpretation, exploratory classification context, limitations, and communication

If the course handout uses different unit names, the same sections can be relabeled to match the official syllabus.

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
- `data/breast_cancer_wdbc.csv` — CSV export of the WDBC dataset used by the analysis
- `report.md` — supporting report with methodology, observations, limitations, and conclusions
- `requirements.txt` — Python dependencies
- `figures/` — generated plots after running the notebook or script

## Visualization resources

The generated image resources are available directly on GitHub:

1. [Diagnostic class distribution](figures/class-distribution.png)
2. [Feature distributions](figures/feature-distributions.png)
3. [Mean-radius boxplot](figures/radius-boxplot.png)
4. [Radius/perimeter scatter plot](figures/radius-perimeter-scatter.png)
5. [Correlation heatmap](figures/correlation-heatmap.png)
6. [PCA projection](figures/pca-projection.png)
7. [Pair plot](figures/pair-plot.png)
8. [Grouped mean-radius chart](figures/grouped-radius-chart.png)
9. [Concavity violin plot](figures/concavity-violin-plot.png)
10. [PCA explained variance](figures/pca-variance.png)
11. [LDA projection](figures/lda-projection.png)
12. [t-SNE projection](figures/tsne-projection.png)
13. [Confusion matrix](figures/confusion-matrix.png)
14. [ROC curves](figures/roc-curves.png)
15. [Feature importance](figures/feature-importance.png)

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

## Unit V lecture implementation

The implementation follows the supplied Unit V lecture topics:

- Pair plots using `seaborn.pairplot`
- Grouped summaries with `groupby` and grouped bar/violin plots
- Correlation heatmap and strongest-pair interpretation
- Standardized PCA, cumulative explained variance, supervised LDA, and t-SNE
- Scikit-learn estimator workflow with stratified splitting and leakage-safe pipelines
- Cross-validation, GridSearchCV, classification report, confusion matrix, ROC-AUC, and feature importance

Additional Unit V figures are saved in `figures/` as files `07` through `15`.

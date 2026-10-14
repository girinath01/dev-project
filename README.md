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

## Visualization resources

The generated image resources are available directly on GitHub:

1. [Diagnostic class distribution](figures/01_class_distribution.png)
2. [Feature distributions by diagnosis](figures/02_feature_distributions.png)
3. [Mean-radius boxplot](figures/03_boxplot_mean_radius.png)
4. [Radius/perimeter scatter plot](figures/04_radius_perimeter_scatter.png)
5. [Correlation heatmap](figures/05_correlation_heatmap.png)
6. [PCA projection](figures/06_pca_projection.png)
7. [Pair plot](figures/07_pairplot_selected_features.png)
8. [Grouped mean-radius chart](figures/08_grouped_mean_radius_ci.png)
9. [Grouped concavity violin plot](figures/09_grouped_violin_concavity.png)
10. [PCA cumulative variance](figures/10_pca_cumulative_variance.png)
11. [LDA projection](figures/11_lda_projection.png)
12. [t-SNE projection](figures/12_tsne_projection.png)
13. [Random-forest confusion matrix](figures/13_random_forest_confusion_matrix.png)
14. [ROC curves](figures/14_roc_curves.png)
15. [Random-forest feature importance](figures/15_random_forest_feature_importance.png)

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

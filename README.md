# Exploratory Data Analysis and Visualization of Cancer Data

## Project overview

This capstone project analyzes a large county-level cancer dataset containing **3,047 U.S. counties**. It studies cancer mortality rates alongside cancer incidence, income, poverty, education, employment, insurance coverage, demographics, and population estimates.

This is an ecological dataset: each row represents a county, not an individual patient. Therefore, the findings describe county-level associations and must not be interpreted as individual medical risk or clinical advice.

## Dataset source

- Kaggle dataset: [Health Outcomes and Socioeconomic Factors](https://www.kaggle.com/datasets/thedevastator/uncovering-trends-in-health-outcomes-and-socioeconomic)
- Dataset description: the original `cancer_reg.csv`-style data contains 3,047 U.S. counties and cancer mortality/incidence outcomes.
- Local copy: `data/cancer_regression.csv`

The local CSV is included so the analysis is reproducible without requiring Kaggle credentials. The analysis uses `TARGET_deathRate` as the continuous target: the county cancer mortality rate.

## Repository contents

- `cancer_patients_eda.py` — complete executable analysis
- `cancer_patients_eda.ipynb` — notebook presentation
- `report.md` — supporting report with methodology, observations, and limitations
- `data/cancer_regression.csv` — local dataset used by the analysis; intentionally excluded from Git
- `requirements.txt` — Python dependencies
- `figures/` — generated visualization resources

## How to run

```bash
python -m pip install -r requirements.txt
python cancer_patients_eda.py
```

The script uses a fixed random seed (`42`) and saves all figures in `figures/`.

## Analysis covered

1. Dataset loading, structure, missing values, duplicates, and data types
2. Descriptive statistics and distributions
3. State-level grouped summaries and boxplots
4. Incidence/mortality relationships and correlation heatmap
5. Pair plot of key variables
6. Standardized PCA and cumulative explained variance
7. t-SNE visualization with a fixed seed
8. Leakage-safe Ridge and Random Forest regression pipelines
9. Five-fold cross-validation and GridSearchCV
10. MAE, RMSE, R², observed-versus-predicted visualization, and feature associations

## Figure resources

- [Mortality distribution](figures/mortality-distribution.png)
- [Key-variable distributions](figures/key-variable-distributions.png)
- [State mortality boxplot](figures/state-mortality-boxplot.png)
- [Incidence/mortality scatter plot](figures/incidence-mortality-scatter.png)
- [Correlation heatmap](figures/correlation-heatmap.png)
- [Pair plot](figures/pair-plot.png)
- [PCA projection](figures/pca-projection.png)
- [PCA explained variance](figures/pca-variance.png)
- [t-SNE projection](figures/tsne-projection.png)
- [Regression predictions](figures/regression-predictions.png)

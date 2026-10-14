# Exploratory Data Analysis and Visualization of Large-Scale Cancer Data

## 1. Abstract

This project analyzes a large county-level cancer dataset containing 3,047 U.S. counties. The dataset combines cancer incidence and mortality measures with socioeconomic, demographic, employment, education, insurance, and population variables. The analysis follows the five-unit Data Exploration and Visualization workflow and applies the Unit V lecture topics: multivariate relationships, grouped visualizations, dimensionality reduction, and scikit-learn model development.

The target variable, `TARGET_deathRate`, is a continuous county cancer mortality rate. The analysis is predictive and exploratory; it does not diagnose patients or estimate individual survival.

## 2. Dataset and source

- **Dataset:** Health Outcomes and Socioeconomic Factors / Cancer Regression dataset
- **Rows:** 3,047 county-level observations
- **Variables:** cancer outcomes, population, income, poverty, education, employment, insurance, demographics, and geography
- **Target:** `TARGET_deathRate`
- **Kaggle source:** [Health Outcomes and Socioeconomic Factors](https://www.kaggle.com/datasets/thedevastator/uncovering-trends-in-health-outcomes-and-socioeconomic)
- **Local file:** `data/cancer_regression.csv`

The dataset is aggregated at county level. It is not a patient-level clinical dataset.

## 3. Objectives

1. Profile a large cancer-related dataset and assess data quality.
2. Describe the distributions of mortality, incidence, and socioeconomic variables.
3. Compare mortality across states and study bivariate relationships.
4. Use correlation analysis, pair plots, PCA, and t-SNE for multivariate exploration.
5. Build and compare regression models using leakage-safe scikit-learn pipelines.
6. Report limitations and avoid unsupported clinical interpretation.

## 4. Methodology

### Unit 1 — Data loading and quality

The CSV is loaded from the repository. An accidental unnamed index column is removed, and the state is extracted from the geography field. The script prints dimensions, sample rows, data types, missing values, duplicate rows, and the number of states represented.

### Unit 2 — Descriptive and univariate analysis

The script calculates count, mean, standard deviation, quartiles, minimum, maximum, and IQR for numeric columns. Histograms and KDE curves show mortality, incidence, median income, and poverty distributions.

### Unit 3 — Bivariate and grouped analysis

State-level count, mean, median, and standard deviation are calculated. A boxplot compares the ten states with the highest mean mortality. A bubble scatter plot compares incidence and mortality, with bubble size representing population and color representing poverty. A masked Pearson correlation heatmap avoids repeating the same pair above and below the diagonal.

### Unit 4 — Multivariate analysis and dimensionality reduction

Numeric features are median-imputed and standardized before PCA and t-SNE. PCA is used for a quick linear projection and cumulative-variance analysis. t-SNE is used only for visual exploration with `perplexity=30` and `random_state=42`. t-SNE distances and cluster sizes must not be interpreted as clinical or geographic effects.

### Unit 5 — Scikit-learn model development

The continuous mortality target is modeled with Ridge regression and Random Forest regression. The data is split before final testing. Imputation, scaling, and one-hot encoding are inside a `ColumnTransformer` pipeline, preventing leakage during cross-validation. Five-fold shuffled cross-validation reports RMSE. `GridSearchCV` tunes Ridge alpha. Test metrics include MAE, RMSE, and R².

## 5. Verified analysis results

The validated run produced:

- Dataset shape: **3,047 rows and 35 columns** after adding the derived state field.
- Two PCA components retain approximately **46.8%** of standardized numeric variance.
- **17 PCA components** are needed to retain at least 95% variance.
- t-SNE mortality-quartile silhouette score: approximately **-0.058**, indicating no strong quartile separation in this embedding.
- Ridge cross-validation RMSE: approximately **21.471**.
- Random Forest cross-validation RMSE: approximately **20.628**.
- Ridge test MAE: approximately **15.179** and test RMSE: approximately **22.436**.
- Random Forest test MAE: approximately **16.668** and test RMSE: approximately **23.526**.

These metrics describe this county-level dataset and split; they are not clinical performance measures.

## 6. Main observations

- Mortality rates vary substantially across counties and states.
- Higher poverty and lower educational attainment show meaningful associations with mortality in this aggregated dataset.
- Incidence and mortality are positively related, but the relationship is not a complete explanation of county mortality differences.
- The first two PCA components provide a useful visual summary but retain less than half of total standardized variance.
- The t-SNE result does not show strong separation by mortality quartile, so a visually attractive embedding should not be over-interpreted.
- Random Forest and Ridge provide complementary views: the forest captures nonlinear patterns, while Ridge is more interpretable and stable for a socioeconomic dataset.

## 7. Limitations and ethics

- County-level relationships cannot be used to infer individual patient outcomes; this is the ecological fallacy risk.
- The dataset is historical and may not represent current populations or healthcare systems.
- Associations do not establish causation.
- Missing values and imputation choices can affect model results.
- The data may contain measurement, reporting, and geographic aggregation bias.
- No result should be used for diagnosis, treatment selection, or individual cancer-risk prediction.

## 8. Reproducibility

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python cancer_patients_eda.py
```

All random operations use seed `42`. Generated figures are stored in `figures/`, and the notebook provides a presentation-friendly entry point.

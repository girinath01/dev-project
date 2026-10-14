"""Exploratory analysis of a large U.S. county-level cancer dataset.

Run with:
    python cancer_patients_eda.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DATASET_PATH = Path("data/cancer_regression.csv")
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def load_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Download the documented source first."
        )
    df = pd.read_csv(DATASET_PATH)
    unnamed = [column for column in df.columns if column.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df["state"] = df["Geography"].str.rsplit(", ", n=1).str[-1]
    return df


def main() -> None:
    # Unit 1: acquisition, structure, and data quality.
    df = load_data()
    target = "TARGET_deathRate"
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [column for column in numeric_features if column != target]

    print("Dataset source:", DATASET_PATH)
    print("Shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head().to_string())
    print("\nData types:")
    print(df.dtypes.value_counts().to_string())
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(10).to_string())
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("States represented:", df["state"].nunique())

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=target, kde=True, color="steelblue")
    plt.axvline(df[target].mean(), color="darkred", linestyle="--", label="Mean")
    plt.title("County cancer mortality rate distribution")
    plt.xlabel("Target cancer mortality rate")
    plt.legend()
    savefig("mortality-distribution.png")

    # Unit 2: descriptive statistics and univariate exploration.
    summary = df[numeric_features].describe().T
    summary["IQR"] = summary["75%"] - summary["25%"]
    print("\nDescriptive statistics:")
    print(summary.round(3).to_string())

    selected = ["TARGET_deathRate", "incidenceRate", "medIncome", "povertyPercent"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feature in zip(axes.flat, selected):
        sns.histplot(data=df, x=feature, kde=True, ax=ax, color="teal")
        ax.set_title(feature)
    fig.suptitle("Distributions of key county-level variables", y=1.02)
    savefig("key-variable-distributions.png")

    # Unit 3: bivariate and grouped analysis.
    print("\nState-level mortality summary:")
    state_summary = (
        df.groupby("state", observed=True)[target]
        .agg(count="count", mean="mean", median="median", std="std")
        .sort_values("mean", ascending=False)
    )
    print(state_summary.head(10).round(2).to_string())

    top_states = state_summary.head(10).index
    grouped_df = df[df["state"].isin(top_states)]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=grouped_df, x="state", y=target, order=top_states, color="lightsteelblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Cancer mortality rate across the ten highest-mean states")
    savefig("state-mortality-boxplot.png")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="incidenceRate",
        y=target,
        size="popEst2015",
        hue="povertyPercent",
        palette="viridis",
        alpha=0.65,
        sizes=(20, 220),
    )
    plt.title("Cancer incidence and mortality by county")
    savefig("incidence-mortality-scatter.png")

    corr_features = [
        "TARGET_deathRate", "incidenceRate", "medIncome", "povertyPercent",
        "PctBachDeg25_Over", "PctUnemployed16_Over", "PctPublicCoverage",
        "PctPrivateCoverage", "PctWhite", "PctBlack", "PctAsian",
    ]
    corr = df[corr_features].corr(method="pearson")
    plt.figure(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag",
                center=0, vmin=-1, vmax=1, square=True)
    plt.title("Correlation heatmap of cancer and socioeconomic variables")
    savefig("correlation-heatmap.png")

    # Unit 5 lecture topic: pair plot and grouped visualization.
    pair_features = ["TARGET_deathRate", "incidenceRate", "medIncome", "povertyPercent"]
    pair_grid = sns.pairplot(
        df[pair_features],
        corner=True,
        diag_kind="kde",
        plot_kws={"alpha": 0.35, "s": 18},
        height=2.0,
    )
    pair_grid.figure.suptitle("Pair plot of key county-level variables", y=1.02)
    pair_grid.figure.savefig(FIGURES / "pair-plot.png", dpi=180, bbox_inches="tight")
    plt.close(pair_grid.figure)

    # Unit 4: PCA, cumulative variance, and t-SNE.
    X_numeric = df[feature_cols]
    X_scaled = SimpleImputer(strategy="median").fit_transform(X_numeric)
    X_scaled = StandardScaler().fit_transform(X_scaled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print("\nPCA explained variance ratios:", np.round(pca.explained_variance_ratio_, 4))
    print("PCA variance retained by two components:",
          round(float(pca.explained_variance_ratio_.sum()), 4))
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df[target],
                    palette="viridis", alpha=0.65, legend=False)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection of county-level cancer data")
    savefig("pca-projection.png")

    pca_full = PCA().fit(X_scaled)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    components_95 = int(np.argmax(cumulative >= 0.95) + 1)
    print("PCA components needed for 95% variance:", components_95)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
    plt.axvline(components_95, color="black", linestyle="--")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.legend()
    savefig("pca-variance.png")

    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=RANDOM_STATE, max_iter=750)
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df[target], cmap="viridis",
                          s=12, alpha=0.7)
    plt.colorbar(scatter, label="Cancer mortality rate")
    plt.title("t-SNE projection colored by mortality rate")
    savefig("tsne-projection.png")
    mortality_bins = pd.qcut(df[target], q=4, labels=False, duplicates="drop")
    print("t-SNE silhouette score by mortality quartile:",
          round(float(silhouette_score(X_tsne, mortality_bins)), 3))

    # Unit 5 lecture topic: leakage-safe regression model development.
    X = df[feature_cols + ["state"]]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    numeric_model_features = feature_cols
    categorical_model_features = ["state"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_model_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_model_features),
        ]
    )
    models = {
        "Ridge regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2
            )),
        ]),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "CV RMSE": -scores.mean(),
            "Test MAE": mean_absolute_error(y_test, predictions),
            "Test RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "Test R2": r2_score(y_test, predictions),
        }
    print("\nRegression model comparison:")
    print(pd.DataFrame(results).T.round(3).to_string())

    tuned_ridge = GridSearchCV(
        models["Ridge regression"],
        {"model__alpha": [0.1, 1, 10, 100, 1000]},
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    tuned_ridge.fit(X_train, y_train)
    print("Best Ridge parameters:", tuned_ridge.best_params_)
    print("Best Ridge CV RMSE:", round(float(-tuned_ridge.best_score_), 3))
    tuned_predictions = tuned_ridge.predict(X_test)
    print("Tuned Ridge test RMSE:",
          round(float(mean_squared_error(y_test, tuned_predictions) ** 0.5), 3))
    print("Tuned Ridge test R2:", round(float(r2_score(y_test, tuned_predictions)), 3))

    forest = models["Random forest"]
    predictions = forest.predict(X_test)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.65)
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(limits, limits, "r--", label="Perfect prediction")
    plt.xlabel("Observed mortality rate")
    plt.ylabel("Predicted mortality rate")
    plt.title("Random forest: observed versus predicted mortality")
    plt.legend()
    savefig("regression-predictions.png")

    # Report the strongest univariate associations for interpretation.
    associations = (
        df[feature_cols + [target]]
        .corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )
    print("\nStrongest feature correlations with mortality rate:")
    print(associations.head(10).round(3).to_string())
    print("\nAnalysis complete. Figures saved to:", FIGURES.resolve())


if __name__ == "__main__":
    main()

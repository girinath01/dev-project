"""Exploratory Data Analysis and Visualization of Cancer Patients Data.

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
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
    silhouette_score,
)
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    # Unit 1: acquisition, structure, and data quality
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
    df = df.drop(columns="target")

    print("Shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes.value_counts())
    print("\nMissing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("\nClass counts:")
    print(df["diagnosis"].value_counts())
    print("\nClass percentages:")
    print((df["diagnosis"].value_counts(normalize=True) * 100).round(2))

    plt.figure(figsize=(7, 5))
    sns.countplot(
        data=df, x="diagnosis", hue="diagnosis", order=["malignant", "benign"],
        palette="Set2", legend=False
    )
    plt.title("Diagnostic class distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of observations")
    savefig("class-distribution.png")

    # Unit 2: descriptive statistics and univariate exploration
    feature_cols = dataset.feature_names.tolist()
    summary = df[feature_cols].describe().T
    summary["IQR"] = summary["75%"] - summary["25%"]
    print("\nDescriptive statistics:")
    print(summary.round(3).to_string())

    selected = ["mean radius", "mean texture", "mean perimeter", "mean area",
                "mean smoothness", "mean concavity"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feature in zip(axes.flat, selected):
        sns.histplot(data=df, x=feature, hue="diagnosis", kde=True,
                     stat="density", common_norm=False, element="step", ax=ax)
        ax.set_title(feature.title())
    fig.suptitle("Feature distributions by diagnosis", y=1.02)
    savefig("feature-distributions.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="diagnosis", y="mean radius", order=["malignant", "benign"])
    sns.stripplot(data=df, x="diagnosis", y="mean radius", order=["malignant", "benign"],
                  color="black", alpha=0.25, size=3)
    plt.title("Mean radius by diagnosis")
    savefig("radius-boxplot.png")

    # Unit 3: bivariate analysis
    group_means = df.groupby("diagnosis")[selected].mean().T
    print("\nGroup means:")
    print(group_means.round(3).to_string())

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="mean radius", y="mean perimeter",
                    hue="diagnosis", style="diagnosis", alpha=0.75, palette="Set1")
    plt.title("Mean radius and mean perimeter")
    savefig("radius-perimeter-scatter.png")

    corr_features = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
    ]
    corr = df[corr_features].corr()
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Correlation matrix for mean measurements")
    savefig("correlation-heatmap.png")

    # Unit 5 lecture topic: pair plots for multiple-variable relationships.
    pair_features = ["mean radius", "mean texture", "mean perimeter",
                     "mean area", "mean concavity", "mean concave points"]
    pair_grid = sns.pairplot(
        df[pair_features + ["diagnosis"]],
        vars=pair_features,
        hue="diagnosis",
        corner=True,
        diag_kind="kde",
        palette="Set2",
        plot_kws={"alpha": 0.65, "s": 25},
        height=1.8,
    )
    pair_grid.figure.suptitle("Pair plot of selected measurements", y=1.02)
    pair_grid.figure.savefig(FIGURES / "pair-plot.png",
                             dpi=180, bbox_inches="tight")
    plt.close(pair_grid.figure)

    # Unit 5 lecture topic: grouped summaries and grouped visualizations.
    grouped = (
        df.groupby("diagnosis", observed=True)["mean radius"]
        .agg(count="count", mean="mean", median="median", std="std")
        .round(3)
    )
    print("\nGrouped mean-radius summary:")
    print(grouped.to_string())
    plt.figure(figsize=(7, 5))
    sns.barplot(data=df, x="diagnosis", y="mean radius", hue="diagnosis",
                order=["malignant", "benign"], palette="Set2",
                errorbar=("ci", 95), capsize=0.1, legend=False)
    plt.title("Mean radius by diagnosis with 95% confidence intervals")
    plt.ylabel("Mean radius")
    savefig("grouped-radius-chart.png")

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="diagnosis", y="mean concavity",
                   hue="diagnosis", order=["malignant", "benign"],
                   palette="Set2", inner="quart", legend=False)
    plt.title("Mean concavity distribution by diagnosis")
    savefig("concavity-violin-plot.png")

    # Unit 5 lecture topic: dimensionality reduction for visualization.
    X = df[feature_cols]
    y = dataset.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                           "diagnosis": df["diagnosis"]})
    print("\nPCA explained variance ratios:", np.round(pca.explained_variance_ratio_, 4))
    print("PCA cumulative variance (2 components):",
          round(float(pca.explained_variance_ratio_.sum()), 4))
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="diagnosis",
                    style="diagnosis", palette="Set1", alpha=0.8)
    plt.title("PCA projection after standardization")
    savefig("pca-projection.png")

    pca_full = PCA().fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    components_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)
    print("PCA components needed for 95% variance:", components_95)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             marker="o", markersize=3)
    plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
    plt.axvline(components_95, color="black", linestyle="--",
                label=f"{components_95} components")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.legend()
    savefig("pca-variance.png")

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X_scaled, y)
    lda_df = pd.DataFrame({"LD1": X_lda[:, 0], "diagnosis": df["diagnosis"]})
    plt.figure(figsize=(8, 4))
    sns.stripplot(data=lda_df, x="LD1", y="diagnosis", hue="diagnosis",
                  order=["malignant", "benign"], palette="Set1",
                  alpha=0.75, legend=False)
    plt.title("LDA supervised projection")
    savefig("lda-projection.png")
    print("LDA explained variance ratio:", np.round(lda.explained_variance_ratio_, 4))

    tsne = TSNE(n_components=2, perplexity=30, init="pca",
                random_state=RANDOM_STATE, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    tsne_df = pd.DataFrame({"TSNE1": X_tsne[:, 0], "TSNE2": X_tsne[:, 1],
                            "diagnosis": df["diagnosis"]})
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=tsne_df, x="TSNE1", y="TSNE2", hue="diagnosis",
                    style="diagnosis", palette="Set1", alpha=0.8)
    plt.title("t-SNE projection (perplexity=30, seed=42)")
    savefig("tsne-projection.png")
    tsne_labels = pd.Categorical(df["diagnosis"]).codes
    print("t-SNE silhouette score (visual diagnostic only):",
          round(float(silhouette_score(X_tsne, tsne_labels)), 3))

    # Unit 5 lecture topic: split first, then fit preprocessing inside pipelines.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        ),
        "SVM": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("\nStratified split:", X_train.shape, X_test.shape)
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"{name}: CV accuracy={scores.mean():.3f} +/- {scores.std():.3f}")
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        print(f"{name}: test ROC-AUC={roc_auc_score(y_test, probabilities):.3f}")

    tuned_svm = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)),
        {
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", 0.01, 0.001],
            "svc__kernel": ["rbf", "linear"],
        },
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    tuned_svm.fit(X_train, y_train)
    print("Best SVM parameters:", tuned_svm.best_params_)
    print("Best SVM CV accuracy:", round(float(tuned_svm.best_score_), 3))
    print("Tuned SVM test accuracy:", round(float(tuned_svm.score(X_test, y_test)), 3))

    best_model = models["Random forest"]
    y_pred = best_model.predict(X_test)
    print("\nRandom forest classification report:")
    print(classification_report(y_test, y_pred, target_names=dataset.target_names))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=dataset.target_names, cmap="Blues"
    )
    plt.title("Random forest confusion matrix")
    savefig("confusion-matrix.png")

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.title("ROC curves")
    plt.legend(loc="lower right")
    savefig("roc-curves.png")

    importance = pd.Series(
        models["Random forest"].feature_importances_, index=feature_cols
    ).sort_values(ascending=False).head(8)
    print("\nTop random-forest feature importances:")
    print(importance.round(3).to_string())
    plt.figure(figsize=(8, 5))
    importance.sort_values().plot.barh(color="teal")
    plt.xlabel("Importance")
    plt.title("Top eight random-forest features")
    savefig("feature-importance.png")

    # Exploratory association ranking, not a clinical risk score.
    top_correlations = (
        df[feature_cols + ["diagnosis"]]
        .assign(diagnosis_code=y)
        .corr(numeric_only=True)["diagnosis_code"]
        .drop("diagnosis_code")
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTen features with largest absolute correlation with diagnosis code:")
    print(top_correlations.round(4).to_string())
    print("\nAnalysis complete. Figures saved to:", FIGURES.resolve())


if __name__ == "__main__":
    main()

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
    savefig("01_class_distribution.png")

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
    savefig("02_feature_distributions.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="diagnosis", y="mean radius", order=["malignant", "benign"])
    sns.stripplot(data=df, x="diagnosis", y="mean radius", order=["malignant", "benign"],
                  color="black", alpha=0.25, size=3)
    plt.title("Mean radius by diagnosis")
    savefig("03_boxplot_mean_radius.png")

    # Unit 3: bivariate analysis
    group_means = df.groupby("diagnosis")[selected].mean().T
    print("\nGroup means:")
    print(group_means.round(3).to_string())

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="mean radius", y="mean perimeter",
                    hue="diagnosis", style="diagnosis", alpha=0.75, palette="Set1")
    plt.title("Mean radius and mean perimeter")
    savefig("04_radius_perimeter_scatter.png")

    corr_features = [
        "mean radius", "mean texture", "mean perimeter", "mean area",
        "mean smoothness", "mean compactness", "mean concavity",
        "mean concave points", "mean symmetry", "mean fractal dimension",
    ]
    corr = df[corr_features].corr()
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Correlation matrix for mean measurements")
    savefig("05_correlation_heatmap.png")

    # Unit 4: multivariate analysis and preprocessing
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
    savefig("06_pca_projection.png")

    # A transparent exploratory split demonstrates leakage-aware preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    train_scaler = StandardScaler().fit(X_train)
    X_train_scaled = train_scaler.transform(X_train)
    X_test_scaled = train_scaler.transform(X_test)
    print("\nStratified split:", X_train.shape, X_test.shape)
    print("Scaled train mean (first 3 features):",
          np.round(X_train_scaled[:, :3].mean(axis=0), 6))
    print("Scaled test shape:", X_test_scaled.shape)

    # Unit 5: interpretation and reproducible outputs
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

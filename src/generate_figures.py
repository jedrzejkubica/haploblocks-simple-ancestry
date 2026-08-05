"""
Generate figures for the ancestry classification analysis:
  1. Confusion matrix
  2. Accuracy vs. number of regions used (scaling curve)
  3. 2D projection (TruncatedSVD) of samples colored by population

Built on top of the already-validated load_individual_hashes /
build_feature_matrix functions from ancestry_model.py.
"""

import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Works whether this script lives in the repo root (next to src/, test_data/)
# or inside src/ itself (next to ancestry_model.py).
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR

spec = importlib.util.spec_from_file_location(
    "ancestry_model", os.path.join(REPO_ROOT, "src", "ancestry_model.py")
)
am = importlib.util.module_from_spec(spec)
spec.loader.exec_module(am)


def figure_confusion_matrix(X, y, feature_regions, out_path):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=100)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Population Classification -- Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()
    print(f"Saved {out_path}")


def figure_accuracy_vs_regions(local_dir, igsr_dir, checkpoints, out_path):
    accuracies = []
    for n in checkpoints:
        long_df = am.load_individual_hashes(local_dir, max_regions=n)
        X, y, feature_regions = am.build_feature_matrix(long_df, igsr_dir)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=0
        )
        clf = RandomForestClassifier(n_estimators=300, random_state=0)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        accuracies.append(acc)
        print(f"  regions={n}: accuracy={acc:.3f}")

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)
    ax.plot(checkpoints, accuracies, marker="o", color="#3498db", linewidth=2, markersize=7)
    ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="Chance level (33%)")
    ax.set_xlabel("Number of haploblock regions used", fontsize=12)
    ax.set_ylabel("Held-out accuracy", fontsize=12)
    ax.set_title("Classification Accuracy vs. Genomic Coverage", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()
    print(f"Saved {out_path}")
    return checkpoints, accuracies


def figure_2d_projection(X, y, out_path):
    svd = TruncatedSVD(n_components=2, random_state=0)
    coords = svd.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    colors = {"CHB": "#e74c3c", "GBR": "#3498db", "PUR": "#2ecc71"}
    for pop in sorted(y.unique()):
        mask = (y == pop).to_numpy()
        ax.scatter(coords[mask, 0], coords[mask, 1], label=pop, alpha=0.7, s=50,
                   color=colors.get(pop, "gray"), edgecolor="white", linewidth=0.5)

    var_explained = svd.explained_variance_ratio_
    ax.set_xlabel("SVD Component 1", fontsize=12)
    ax.set_ylabel("SVD Component 2", fontsize=12)
    ax.set_title("Samples Projected by Haploblock Cluster Assignment", fontsize=13, fontweight="bold", pad=15)
    ax.legend(title="Population")
    ax.grid(True, linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.text(0.5, -0.02,
              "Note: with 169k+ sparse features, % variance explained by 2 components is tiny by construction\n"
              "(these axes are the best 2D separating directions found, not a variance-proportion summary).",
              ha="center", fontsize=8, style="italic", color="gray")
    plt.tight_layout()
    plt.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path} (component variance ratios: {var_explained[0]:.2e}, {var_explained[1]:.2e} -- expected to be tiny at this dimensionality, see note on figure)")


if __name__ == "__main__":
    local_dir = os.path.join(REPO_ROOT, "test_data", "hashes")
    igsr_dir = os.path.join(REPO_ROOT, "test_data", "igsr")

    long_df = am.load_individual_hashes(local_dir, max_regions=None)
    X, y, feature_regions = am.build_feature_matrix(long_df, igsr_dir)

    figures_dir = os.path.join(REPO_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    figure_confusion_matrix(X, y, feature_regions, os.path.join(figures_dir, "confusion_matrix.png"))
    figure_2d_projection(X, y, os.path.join(figures_dir, "population_projection.png"))
    figure_accuracy_vs_regions(
        local_dir, igsr_dir,
        checkpoints=[10, 50, 100, 250, 500, 1000, 2287],
        out_path=os.path.join(figures_dir, "accuracy_vs_regions.png"),
    )

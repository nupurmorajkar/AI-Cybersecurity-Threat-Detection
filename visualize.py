# src/visualize.py
# Generates all visualization graphs and saves them to images/

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

os.makedirs("images", exist_ok=True)

def plot_label_distribution(y_train):
    """Bar chart showing normal vs attack ratio."""
    counts = y_train.value_counts()
    labels = ["Normal", "Attack"]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=labels, y=counts.values, palette=["steelblue", "crimson"])
    plt.title("Traffic Label Distribution (Train Set)", fontsize=14)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("images/label_distribution.png")
    plt.close()
    print("Saved: images/label_distribution.png")


def plot_confusion_matrix(cm):
    """Heatmap of the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png")
    plt.close()
    print("Saved: images/confusion_matrix.png")


def plot_feature_importance(model, feature_names):
    """Top 15 most important features according to the model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # top 15

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        palette="viridis"
    )
    plt.title("Top 15 Feature Importances", fontsize=14)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png")
    plt.close()
    print("Saved: images/feature_importance.png")


def plot_anomaly_scores(model, X_test, y_test):
    """
    Shows predicted probabilities for a sample of test records.
    Higher probability = model is more confident it's an attack.
    """
    proba = model.predict_proba(X_test)[:, 1]  # probability of being attack

    # Take first 200 samples for readability
    sample_proba = proba[:200]
    sample_labels = list(y_test)[:200]

    plt.figure(figsize=(14, 5))
    colors = ["crimson" if l == 1 else "steelblue" for l in sample_labels]
    plt.bar(range(len(sample_proba)), sample_proba, color=colors, width=1.0)
    plt.axhline(y=0.5, color="black", linestyle="--", label="Decision threshold (0.5)")
    plt.title("Anomaly Score per Sample (Red = Attack, Blue = Normal)", fontsize=13)
    plt.xlabel("Sample Index")
    plt.ylabel("Attack Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/anomaly_scores.png")
    plt.close()
    print("Saved: images/anomaly_scores.png")
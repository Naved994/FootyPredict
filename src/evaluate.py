"""
FootyPredict: Evaluation & Visualization Suite
Generates high-resolution academic charts for the Mini Project Report:
1. Confusion Matrix
2. Model Comparison Bar Chart (Accuracy, Precision, Recall, F1-Score)
3. Feature Importance Ranking
4. Bivariate Scoreline Probability Heatmap
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "evaluation_results.json")

def set_academic_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = "#e2e8f0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7

def plot_confusion_matrix(cm, classes, output_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True Match Result',
        xlabel='Predicted Match Result',
        title='Normalized Confusion Matrix — FootyPredict'
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_pct = f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})"
            ax.text(j, i, val_pct,
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "#0f172a",
                    fontsize=9, fontweight="bold")

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Confusion Matrix -> {output_path}")

def plot_model_comparison(metrics_summary, output_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    models = list(metrics_summary.keys())
    accuracies = [metrics_summary[m]["Accuracy"] for m in models]
    f1_macros = [metrics_summary[m]["F1_Macro"] for m in models]
    f1_weights = [metrics_summary[m]["F1_Weighted"] for m in models]

    x = np.arange(len(models))
    width = 0.26

    rects1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color='#0284c7', edgecolor='#0369a1')
    rects2 = ax.bar(x, f1_weights, width, label='F1-Score Weighted (%)', color='#10b981', edgecolor='#059669')
    rects3 = ax.bar(x + width, f1_macros, width, label='F1-Score Macro (%)', color='#8b5cf6', edgecolor='#7c3aed')

    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold', color='#1e293b')
    ax.set_title('Academic Benchmark: Model Outcome Comparison on 3-Class Test Split', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12, ha='right', fontsize=9, fontweight='semibold')
    ax.set_ylim(0, 65)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    ax.grid(axis='y', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Model Comparison -> {output_path}")

def plot_feature_importance(feature_importances, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    sorted_feats = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:10]
    names = [k for k, _ in sorted_feats][::-1]
    scores = [v for _, v in sorted_feats][::-1]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color='#0ea5e9', edgecolor='#0284c7', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, fontweight='semibold')
    ax.set_xlabel('Relative GBDT Feature Importance', fontsize=10, fontweight='bold')
    ax.set_title('Top 10 Match Outcome Predictive Features', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.5)

    for i, v in enumerate(scores):
        ax.text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=8, fontweight='bold', color='#1e293b')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Feature Importance -> {output_path}")

def plot_scoreline_heatmap(output_path):
    from model import DualStageFootballPredictor
    lam_h = 2.15  # Home attack expectancy
    lam_a = 1.10  # Away attack expectancy
    mat = DualStageFootballPredictor.compute_scoreline_matrix(lam_h, lam_a, max_goals=5)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(mat * 100, cmap='YlGnBu')
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"Away {i}" for i in range(5)], fontsize=9)
    ax.set_yticklabels([f"Home {i}" for i in range(5)], fontsize=9)
    ax.set_xlabel('Away Goals (Chelsea: λ = 1.10)', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel('Home Goals (Arsenal: λ = 2.15)', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title('Bivariate Poisson Score Probability Matrix (%)', fontsize=11, fontweight='bold', pad=15)

    thresh = (mat * 100).max() / 2.
    for x in range(5):
        for y in range(5):
            pct = mat[x, y] * 100
            ax.text(y, x, f"{pct:.1f}%", ha='center', va='center',
                    color="white" if pct > thresh else "#0f172a",
                    fontsize=8.5, fontweight='bold')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved Scoreline Heatmap -> {output_path}")

def run_evaluation_suite():
    os.makedirs(FIGS_DIR, exist_ok=True)
    set_academic_style()

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    classes = data["classes"]
    metrics_summary = data["metrics_summary"]
    feature_importances = data["feature_importances"]

    plot_confusion_matrix(cm, classes, os.path.join(FIGS_DIR, "confusion_matrix.png"))
    plot_model_comparison(metrics_summary, os.path.join(FIGS_DIR, "model_comparison.png"))
    plot_feature_importance(feature_importances, os.path.join(FIGS_DIR, "feature_importance.png"))
    plot_scoreline_heatmap(os.path.join(FIGS_DIR, "scoreline_heatmap.png"))
    print("\n[COMPLETE] All academic evaluation figures generated successfully.")

if __name__ == "__main__":
    run_evaluation_suite()

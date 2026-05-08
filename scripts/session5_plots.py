"""
Session 5 — Génération des plots GitHub
Projet : Prédiction de l'endométriose
Auteur : Noa Attia

Génère 3 plots dans le dossier plots/ :
  01_eda_correlation.png         — heatmap de corrélation (EDA)
  02_model_comparison.png        — comparaison des 3 modèles (F1, accuracy, recall)
  03_best_model_results.png      — confusion matrix + ROC du meilleur modèle
"""

import os
import sys
import pickle
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)

from src.data import load_dataset_split

# ─── Setup ───────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")
PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb"]

print("🎨 Génération des plots Session 5...")

# ─── Chargement ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = load_dataset_split()

# ─── 1. EDA — Heatmap de corrélation ─────────────────────────
df_raw = pd.read_csv("data/structured_endometriosis_data.csv")

fig, ax = plt.subplots(figsize=(9, 7))
corr = df_raw.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, annot_kws={"size": 11})
ax.set_title("Matrice de corrélation — Features cliniques",
             fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("plots/01_eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ plots/01_eda_correlation.png")

# ─── 2. Comparaison des modèles ──────────────────────────────
models_info = {
    "Logistic\nRegression": "models/logistic_regression.pkl",
    "Random\nForest":       "models/random_forest.pkl",
    "Gradient\nBoosting":   "models/gradient_boosting.pkl",
}

metrics_data = {"model": [], "F1-score": [], "Accuracy": [], "Recall": []}
from src.metrics import compute_metrics

for label, path in models_info.items():
    with open(path, "rb") as f:
        clf = pickle.load(f)
    y_pred = clf.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    metrics_data["model"].append(label)
    metrics_data["F1-score"].append(m["f1"])
    metrics_data["Accuracy"].append(m["accuracy"])
    metrics_data["Recall"].append(m["recall"])

df_metrics = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df_metrics))
width = 0.25

bars1 = ax.bar(x - width, df_metrics["F1-score"], width, label="F1-score",
               color=PALETTE[0], edgecolor="white")
bars2 = ax.bar(x,          df_metrics["Accuracy"], width, label="Accuracy",
               color=PALETTE[1], edgecolor="white")
bars3 = ax.bar(x + width,  df_metrics["Recall"],   width, label="Recall",
               color=PALETTE[2], edgecolor="white")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(df_metrics["model"], fontsize=12)
ax.set_ylim(0.80, 1.00)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Comparaison des modèles — F1, Accuracy, Recall",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("plots/02_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ plots/02_model_comparison.png")

# ─── 3. Meilleur modèle : Confusion Matrix + Courbe ROC ──────
with open("models/gradient_boosting.pkl", "rb") as f:
    best_model = pickle.load(f)

y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba_best)
roc_auc_val = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Gradient Boosting — Meilleur modèle", fontsize=14, fontweight="bold")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Non (0)", "Oui (1)"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Matrice de confusion", fontsize=13)

# ROC curve
axes[1].plot(fpr, tpr, color="#fc8d62", lw=2.5,
             label=f"AUC = {roc_auc_val:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
axes[1].fill_between(fpr, tpr, alpha=0.1, color="#fc8d62")
axes[1].set_xlabel("Taux de faux positifs", fontsize=12)
axes[1].set_ylabel("Taux de vrais positifs", fontsize=12)
axes[1].set_title("Courbe ROC", fontsize=13)
axes[1].legend(fontsize=12, loc="lower right")
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig("plots/03_best_model_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ plots/03_best_model_results.png")

print("\n🎉 Tous les plots générés dans plots/")

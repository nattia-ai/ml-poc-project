"""
Session 3 — EDA visuelle
Projet : Prédiction de l'endométriose
Auteur : Noa Attia

Génère les visualisations dans le dossier deliverables/plots/
"""

import os
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # mode sans écran
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import add_features, TARGET

# ─── Setup ───────────────────────────────────────────────────
os.makedirs("deliverables/plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="Set2")

df = pd.read_csv("data/structured_endometriosis_data.csv")
df_fe = add_features(df)

print("🎨 Génération des visualisations Session 3...")

# ─── 1. Histogrammes de toutes les features ──────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Distribution des features", fontsize=14, fontweight="bold")
axes = axes.flatten()
cols = ["Age", "Chronic_Pain_Level", "BMI", "Menstrual_Irregularity",
        "Hormone_Level_Abnormality", "Infertility", "Diagnosis"]
for i, col in enumerate(cols):
    axes[i].hist(df[col], bins=20, edgecolor="white", color=sns.color_palette("Set2")[i % 8])
    axes[i].set_title(col)
    axes[i].set_xlabel("")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig("deliverables/plots/01_distributions.png", dpi=150)
plt.close()
print("  ✅ 01_distributions.png")

# ─── 2. Distribution par classe (target) ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Distribution des features numériques par diagnostic", fontsize=13)
for i, col in enumerate(["Age", "Chronic_Pain_Level", "BMI"]):
    for label, grp in df.groupby("Diagnosis"):
        axes[i].hist(grp[col], bins=25, alpha=0.6,
                     label=f"Diagnosis={label}", edgecolor="white")
    axes[i].set_title(col)
    axes[i].legend()
plt.tight_layout()
plt.savefig("deliverables/plots/02_features_by_target.png", dpi=150)
plt.close()
print("  ✅ 02_features_by_target.png")

# ─── 3. Heatmap de corrélation ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title("Matrice de corrélation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("deliverables/plots/03_correlation_heatmap.png", dpi=150)
plt.close()
print("  ✅ 03_correlation_heatmap.png")

# ─── 4. Class imbalance ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = df["Diagnosis"].value_counts()
bars = ax.bar(["Pas d'endométriose (0)", "Endométriose (1)"],
              counts.values, color=["#66c2a5", "#fc8d62"], edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=11)
ax.set_title("Répartition de la variable cible (Diagnosis)", fontweight="bold")
ax.set_ylabel("Nombre de cas")
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig("deliverables/plots/04_class_imbalance.png", dpi=150)
plt.close()
print("  ✅ 04_class_imbalance.png")

# ─── 5. Nouvelles features (Risk_Score & High_Pain) ──────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Nouvelles features — Feature Engineering", fontsize=13)

# Risk Score par diagnostic
sns.countplot(data=df_fe, x="Risk_Score", hue="Diagnosis",
              palette={0: "#66c2a5", 1: "#fc8d62"}, ax=axes[0])
axes[0].set_title("Risk Score vs Diagnosis")
axes[0].legend(title="Diagnosis", labels=["Non (0)", "Oui (1)"])

# High Pain par diagnostic
sns.countplot(data=df_fe, x="High_Pain", hue="Diagnosis",
              palette={0: "#66c2a5", 1: "#fc8d62"}, ax=axes[1])
axes[1].set_title("High Pain vs Diagnosis")
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["Douleur faible (0)", "Douleur sévère (1)"])
axes[1].legend(title="Diagnosis", labels=["Non (0)", "Oui (1)"])

plt.tight_layout()
plt.savefig("deliverables/plots/05_new_features.png", dpi=150)
plt.close()
print("  ✅ 05_new_features.png")

print("\n🎉 Tous les graphiques générés dans deliverables/plots/")

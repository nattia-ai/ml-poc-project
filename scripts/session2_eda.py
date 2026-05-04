"""
Session 2 — Checks qualité du dataset
Projet : Prédiction de l'endométriose
Auteur : Noa Attia
"""

import pandas as pd
import numpy as np

# ─── Chargement ──────────────────────────────────────────────
df = pd.read_csv("data/structured_endometriosis_data.csv")

print("=" * 60)
print("SESSION 2 — CHECKS QUALITÉ DU DATASET")
print("Sujet : Prédiction de l'endométriose")
print("=" * 60)

# ─── 1. Aperçu général ───────────────────────────────────────
print("\n📋 1. APERÇU GÉNÉRAL")
print(f"  Lignes : {df.shape[0]} | Colonnes : {df.shape[1]}")
print("\nTypes de colonnes :")
print(df.dtypes)
print("\ndf.describe() :")
print(df.describe().round(2))

# ─── 2. Valeurs manquantes ───────────────────────────────────
print("\n" + "=" * 60)
print("🔍 2. VALEURS MANQUANTES")

print("\nNombre par colonne (df.isna().sum()) :")
print(df.isna().sum())

print("\nProportion par colonne (df.isna().mean()) :")
print(df.isna().mean().round(4))

cols_to_drop = df.columns[df.isna().mean() > 0.05].tolist()
if cols_to_drop:
    print(f"\n⚠️  Colonnes avec >5% de manquants (à supprimer) : {cols_to_drop}")
else:
    print("\n✅ Aucune colonne avec plus de 5% de valeurs manquantes.")

# ─── 3. Outliers (méthode IQR) ───────────────────────────────
print("\n" + "=" * 60)
print("📊 3. DÉTECTION DES OUTLIERS (méthode IQR)")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    pct = n_out / len(df) * 100
    flag = "⚠️" if pct > 5 else "✅"
    print(f"  {flag} {col:35s}: {n_out:4d} outliers ({pct:.2f}%)")

# ─── 4. Feature drift ────────────────────────────────────────
print("\n" + "=" * 60)
print("📈 4. FEATURE DRIFT (stabilité sur l'ensemble du dataset)")

binary_cols = ["Menstrual_Irregularity", "Hormone_Level_Abnormality",
               "Infertility", "Diagnosis"]
n_splits = 5
for col in binary_cols:
    splits = np.array_split(df[col], n_splits)
    means = [round(s.mean(), 3) for s in splits]
    drift = max(means) - min(means)
    flag = "⚠️  DRIFT" if drift > 0.05 else "✅ stable"
    print(f"  {col:35s}: variation={drift:.3f}  {flag}")

# ─── 5. Class imbalance ──────────────────────────────────────
print("\n" + "=" * 60)
print("⚖️  5. CLASS IMBALANCE (target : Diagnosis)")

print(df["Diagnosis"].value_counts())
print()
print(df["Diagnosis"].value_counts(normalize=True).round(4))

ratio = df["Diagnosis"].value_counts().max() / df["Diagnosis"].value_counts().min()
if ratio > 3:
    print(f"\n⚠️  Déséquilibre important (ratio {ratio:.1f}x)")
    print("   → Utiliser class_weight='balanced' ou SMOTE en Session 3+")
else:
    print(f"\n✅ Classes équilibrées (ratio {ratio:.1f}x)")

# ─── 6. Corrélations avec la target ──────────────────────────
print("\n" + "=" * 60)
print("🔗 6. CORRÉLATIONS AVEC LA TARGET (Diagnosis)")

corr = df.corr()["Diagnosis"].drop("Diagnosis").sort_values(ascending=False)
print(corr.round(3))

print("\n" + "=" * 60)
print("✅ SESSION 2 TERMINÉE")
print("=" * 60)

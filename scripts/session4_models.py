"""
Session 4 — Entraînement des modèles ML
Projet : Prédiction de l'endométriose
Auteur : Noa Attia

Modèles entraînés :
  1. Logistic Regression     (baseline linéaire)
  2. Random Forest           (ensemble bagging)
  3. Gradient Boosting       (ensemble boosting)

Chaque modèle est sauvegardé en .pkl dans le dossier models/.
Le scaler StandardScaler est aussi sauvegardé (models/scaler.pkl)
pour être réutilisé dans la démo Streamlit.
"""

import os
import sys
import pickle
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.data import load_dataset_split
from src.features import NUMERIC_FEATURES
from src.metrics import compute_metrics

# ─── Setup ───────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("SESSION 4 — ENTRAÎNEMENT DES MODÈLES")
print("Sujet : Prédiction de l'endométriose")
print("=" * 60)

# ─── Chargement des données ──────────────────────────────────
print("\n📂 Chargement et préparation des données...")
X_train, X_test, y_train, y_test = load_dataset_split()

print(f"\n  Train : {X_train.shape} | Test : {X_test.shape}")
print(f"  Features : {list(X_train.columns)}")

# ─── Sauvegarde du scaler (fitté dans load_dataset_split) ────
# On re-fitte un scaler sur X_train pour pouvoir le sauvegarder
# et l'utiliser dans la démo Streamlit avec de vraies entrées utilisateur
scaler = StandardScaler()
num_cols = [c for c in NUMERIC_FEATURES if c in X_train.columns]
scaler.fit(X_train[num_cols])

with open("models/scaler.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "numeric_cols": num_cols}, f)
print(f"\n  ✅ Scaler sauvegardé : models/scaler.pkl")
print(f"     Colonnes scalées : {num_cols}")

# ─── Définition des modèles ──────────────────────────────────
models = {
    "logistic_regression": {
        "name": "Logistic Regression (baseline)",
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # compense le déséquilibre 75/25
            random_state=42,
            solver="lbfgs",
        ),
    },
    "random_forest": {
        "name": "Random Forest",
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "model": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
    },
}

# ─── Entraînement et évaluation ──────────────────────────────
results = []

for model_key, model_info in models.items():
    print(f"\n{'─'*60}")
    print(f"🔧 {model_info['name']}")

    clf = model_info["model"]

    # Cross-validation (5 folds) sur le train set
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
    print(f"  CV F1 (5 folds) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Entraînement sur tout le train set
    clf.fit(X_train, y_train)

    # Prédictions
    y_pred = clf.predict(X_test)

    # Métriques
    metrics = compute_metrics(y_test, y_pred)
    results.append({
        "model": model_info["name"],
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4),
        **metrics
    })

    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")

    print("\n  Rapport de classification détaillé :")
    print(classification_report(y_test, y_pred,
                                 target_names=["Non (0)", "Oui (1)"]))

    # Sauvegarde du modèle
    model_path = f"models/{model_key}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"  ✅ Modèle sauvegardé : {model_path}")

# ─── Tableau récapitulatif ───────────────────────────────────
print(f"\n{'='*60}")
print("📊 RÉCAPITULATIF DES PERFORMANCES")
print(f"{'='*60}")
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("f1", ascending=False).reset_index(drop=True)
print(df_results.to_string(index=False))

best = df_results.iloc[0]["model"]
print(f"\n🏆 Meilleur modèle (F1-score) : {best}")

print(f"\n{'='*60}")
print("✅ SESSION 4 TERMINÉE")
print(f"{'='*60}")

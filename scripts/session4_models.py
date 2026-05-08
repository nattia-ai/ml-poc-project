"""
Session 4 — Entraînement des modèles ML
Projet : Prédiction de l'endométriose
Auteur : Noa Attia

Modèles entraînés :
  1. Logistic Regression     (baseline linéaire)
  2. Random Forest           (ensemble bagging)
  3. Gradient Boosting       (ensemble boosting)

Chaque modèle est sauvegardé en .pkl dans le dossier models/.
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

from src.data import load_dataset_split
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

    # Entraînement
    clf.fit(X_train, y_train)

    # Prédictions
    y_pred = clf.predict(X_test)

    # Métriques
    metrics = compute_metrics(y_test, y_pred)
    results.append({"model": model_info["name"], **metrics})

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

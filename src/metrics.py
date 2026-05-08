"""Student-owned metrics contract.

Session 4 — Évaluation des modèles
Projet : Prédiction de l'endométriose
Auteur : Noa Attia
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calcule les métriques d'évaluation pour la classification binaire.

    Métriques retournées :
    - accuracy  : taux de bonnes prédictions global
    - precision : parmi les cas prédits positifs, combien sont réellement positifs
    - recall    : parmi les vrais positifs, combien sont détectés (sensibilité)
    - f1        : moyenne harmonique precision/recall (robuste à l'imbalance)
    - roc_auc   : aire sous la courbe ROC

    Le F1-score est la métrique principale car le dataset est déséquilibré (75/25).
    """
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_pred)),
    }

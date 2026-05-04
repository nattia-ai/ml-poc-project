"""
Session 3 — Feature engineering & encodage
Projet : Prédiction de l'endométriose
Auteur : Noa Attia
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─── Features numériques et binaires ─────────────────────────
NUMERIC_FEATURES = ["Age", "Chronic_Pain_Level", "BMI"]
BINARY_FEATURES  = ["Menstrual_Irregularity", "Hormone_Level_Abnormality", "Infertility"]
TARGET           = "Diagnosis"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute de nouvelles features au dataset.

    Nouvelles features créées :
    - High_Pain       : douleur sévère (Chronic_Pain_Level > 5)
    - Risk_Score      : somme des facteurs de risque binaires
    - Age_Group       : tranche d'âge (young / mid / senior)
    """
    df = df.copy()

    # Feature 1 : douleur sévère (seuil clinique à 5/10)
    df["High_Pain"] = (df["Chronic_Pain_Level"] > 5).astype(int)

    # Feature 2 : score de risque cumulé (0 à 3)
    df["Risk_Score"] = (
        df["Menstrual_Irregularity"]
        + df["Hormone_Level_Abnormality"]
        + df["Infertility"]
    )

    # Feature 3 : tranche d'âge (one-hot encoding via pd.get_dummies)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 100],
        labels=["young", "mid", "senior"]
    )
    df = pd.get_dummies(df, columns=["Age_Group"], prefix="Age", drop_first=False)
    # Conversion booléens -> int
    for col in [c for c in df.columns if c.startswith("Age_")]:
        df[col] = df[col].astype(int)

    return df


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique le StandardScaler sur les features numériques.

    Le scaler est fitté UNIQUEMENT sur X_train pour éviter le data leakage.
    La même transformation est appliquée à X_test.
    """
    scaler = StandardScaler()

    # Colonnes numériques présentes dans X_train
    num_cols = [c for c in NUMERIC_FEATURES if c in X_train.columns]

    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])   # transform seulement !

    return X_train, X_test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des colonnes à utiliser comme features (hors target)."""
    return [c for c in df.columns if c != TARGET]

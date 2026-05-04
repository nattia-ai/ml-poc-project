"""
Session 3 — Feature engineering & encodage
Projet : Prédiction de l'endométriose
Auteur : Noa Attia
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skrub import TableVectorizer


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

    num_cols = [c for c in NUMERIC_FEATURES if c in X_train.columns]

    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])   # transform seulement !

    return X_train, X_test


def apply_skrub(X_train: pd.DataFrame,
                X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique skrub TableVectorizer pour encoder automatiquement toutes les colonnes.

    TableVectorizer détecte automatiquement le type de chaque colonne et applique
    l'encodage approprié (StandardScaler pour le numérique, OneHotEncoder pour le
    catégoriel). Fitté uniquement sur X_train.
    """
    vectorizer = TableVectorizer()

    X_train_skrub = pd.DataFrame(
        vectorizer.fit_transform(X_train),
        columns=vectorizer.get_feature_names_out(),
        index=X_train.index
    )
    X_test_skrub = pd.DataFrame(
        vectorizer.transform(X_test),
        columns=vectorizer.get_feature_names_out(),
        index=X_test.index
    )

    return X_train_skrub, X_test_skrub


def apply_pca(X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              variance_threshold: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Applique une PCA pour réduire la dimensionnalité.

    ⚠️  Les features doivent être standardisées avant d'appliquer la PCA.
    Le nombre de composantes est choisi automatiquement pour conserver
    `variance_threshold` (défaut 95%) de la variance expliquée.

    Args:
        X_train: features d'entraînement (standardisées)
        X_test:  features de test (standardisées)
        variance_threshold: part de variance à conserver (0-1)

    Returns:
        (X_train_pca, X_test_pca, pca_model)
    """
    pca = PCA(n_components=variance_threshold, random_state=42)

    X_train_pca = pd.DataFrame(
        pca.fit_transform(X_train),
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=X_train.index
    )
    X_test_pca = pd.DataFrame(
        pca.transform(X_test),
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=X_test.index
    )

    print(f"  PCA : {X_train.shape[1]} features → {pca.n_components_} composantes")
    print(f"  Variance expliquée cumulée : {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

    return X_train_pca, X_test_pca, pca


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des colonnes à utiliser comme features (hors target)."""
    return [c for c in df.columns if c != TARGET]

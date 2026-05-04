"""Student-owned dataset loading contract.

Session 3 — Train/test split + transformations
Projet : Prédiction de l'endométriose
Auteur : Noa Attia
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import add_features, scale_features, get_feature_columns, TARGET

DATA_PATH = "data/structured_endometriosis_data.csv"
TEST_SIZE  = 0.2
RANDOM_STATE = 42


def load_dataset_split() -> tuple:
    """Charge, transforme et divise le dataset.

    Pipeline complet :
    1. Chargement du CSV
    2. Feature engineering (nouvelles features)
    3. Train/test split (80/20, stratifié sur la target)
    4. StandardScaling des features numériques (fitté sur train seulement)

    Returns:
        (X_train, X_test, y_train, y_test)
    """

    # ── 1. Chargement ─────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)

    # ── 2. Feature engineering ────────────────────────────────
    df = add_features(df)

    # ── 3. Séparation features / target ───────────────────────
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET]

    # ── 4. Train / test split (stratifié pour respecter l'imbalance) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # preserve le ratio 75/25 dans les deux splits
    )

    # ── 5. Scaling (fit sur train, transform sur les deux) ────
    X_train, X_test = scale_features(X_train, X_test)

    print(f"✅ Dataset chargé : {len(df)} lignes | {len(feature_cols)} features")
    print(f"   Train : {X_train.shape} | Test : {X_test.shape}")
    print(f"   Distribution target (train) :\n{y_train.value_counts(normalize=True).round(3)}")

    return X_train, X_test, y_train, y_test

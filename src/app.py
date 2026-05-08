"""
Session 5 — Dashboard Streamlit
Projet : Prédiction de l'endométriose
Auteur : Noa Attia

3 sections :
  1. Problème métier & EDA
  2. Comparaison des modèles
  3. Démo interactive — prédiction en temps réel
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt

from config import MODEL_METRICS_FILE, PROJECT_ROOT, MODELS_DIR

# ─── Helpers ─────────────────────────────────────────────────
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "structured_endometriosis_data.csv")


@st.cache_resource
def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler():
    path = MODELS_DIR / "scaler.pkl"
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["scaler"], obj["numeric_cols"]


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    if MODEL_METRICS_FILE.exists():
        return pd.read_csv(MODEL_METRICS_FILE)
    return None


# ─── App ─────────────────────────────────────────────────────
def build_app() -> None:
    st.set_page_config(
        page_title="Endométriose — ML Dashboard",
        page_icon="🔬",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Endometriosis_Anatomy.png/220px-Endometriosis_Anatomy.png",
            use_column_width=True,
        )
        st.markdown("## Navigation")
        section = st.radio(
            "Choisir une section",
            ["🏥 Problème & EDA", "📊 Comparaison des modèles", "🤖 Démo interactive"],
        )
        st.markdown("---")
        st.markdown("**Auteur :** Noa Attia  \n**Projet :** ML Proof of Concept")

    # ── Section 1 : Problème & EDA ────────────────────────────
    if section == "🏥 Problème & EDA":
        st.title("🔬 Prédiction de l'endométriose")
        st.markdown(
            """
            ### Problème métier
            L'endométriose est une maladie chronique touchant **~10 % des femmes** en âge de procréer,
            avec un délai de diagnostic moyen de **7 à 10 ans**.

            L'objectif est de construire un outil d'aide à la décision médicale capable de **prédire
            si une patiente est atteinte d'endométriose** à partir de ses symptômes cliniques, afin
            d'orienter plus rapidement vers les examens complémentaires.

            > **Type de problème :** Classification binaire
            > **Target :** `Diagnosis` (0 = non, 1 = endométriose)
            """
        )

        st.markdown("---")
        st.subheader("📋 Aperçu du dataset")
        df = load_raw_data()
        col1, col2, col3 = st.columns(3)
        col1.metric("Lignes", f"{len(df):,}")
        col2.metric("Features", str(df.shape[1] - 1))
        col3.metric("% positifs", f"{df['Diagnosis'].mean()*100:.1f} %")

        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.subheader("📈 Analyses exploratoires")

        tab1, tab2, tab3 = st.tabs(["Distribution des features", "Déséquilibre de classes", "Corrélations"])

        with tab1:
            plot_path = PROJECT_ROOT / "deliverables" / "plots" / "02_features_by_target.png"
            if plot_path.exists():
                st.image(str(plot_path), use_column_width=True)
            else:
                st.info("Lancer `python scripts/session3_eda.py` pour générer les plots.")

        with tab2:
            plot_path = PROJECT_ROOT / "deliverables" / "plots" / "04_class_imbalance.png"
            if plot_path.exists():
                st.image(str(plot_path), use_column_width=True)
            st.warning(
                "⚠️ **Déséquilibre important** : 75 % positifs / 25 % négatifs (ratio 3.1x).  \n"
                "→ Tous les modèles utilisent `class_weight='balanced'`."
            )

        with tab3:
            plot_path = PROJECT_ROOT / "plots" / "01_eda_correlation.png"
            if plot_path.exists():
                st.image(str(plot_path), use_column_width=True)
            st.info(
                "**Features les plus corrélées à la target :**  \n"
                "- `Menstrual_Irregularity` : 0.70  \n"
                "- `Chronic_Pain_Level` : 0.51"
            )

    # ── Section 2 : Comparaison des modèles ──────────────────
    elif section == "📊 Comparaison des modèles":
        st.title("📊 Comparaison des modèles ML")

        st.markdown(
            """
            Trois modèles ont été entraînés sur 80 % du dataset (stratifié) et évalués sur les 20 % restants.
            Le **F1-score** est la métrique principale en raison du déséquilibre de classes.

            | Modèle | Type | Hyperparamètres clés |
            |---|---|---|
            | Logistic Regression | Baseline linéaire | `max_iter=1000`, `class_weight=balanced` |
            | Random Forest | Ensemble (bagging) | `n_estimators=200`, `max_depth=10` |
            | Gradient Boosting | Ensemble (boosting) | `n_estimators=200`, `lr=0.05`, `max_depth=4` |
            """
        )

        st.markdown("---")
        st.subheader("📈 Graphique de comparaison")
        comparison_plot = PROJECT_ROOT / "plots" / "02_model_comparison.png"
        if comparison_plot.exists():
            st.image(str(comparison_plot), use_column_width=True)

        st.markdown("---")
        st.subheader("📋 Tableau des métriques")
        metrics_df = load_metrics()
        if metrics_df is not None:
            display_df = metrics_df[["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
            display_df.columns = ["Modèle", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
            display_df = display_df.sort_values("F1-score", ascending=False).reset_index(drop=True)

            # Mise en forme
            styled = display_df.style.format({
                "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                "Recall": "{:.4f}", "F1-score": "{:.4f}", "ROC-AUC": "{:.4f}",
            }).highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
                color="#d4edda",
            )
            st.dataframe(styled, use_container_width=True)

            best_model = display_df.iloc[0]["Modèle"]
            best_f1 = display_df.iloc[0]["F1-score"]
            st.success(f"🏆 **Meilleur modèle : {best_model}** — F1-score = {best_f1:.4f}")
        else:
            st.info("Lancer `python scripts/main.py` pour générer les métriques.")

        st.markdown("---")
        st.subheader("🎯 Résultats du meilleur modèle (Gradient Boosting)")
        best_plot = PROJECT_ROOT / "plots" / "03_best_model_results.png"
        if best_plot.exists():
            st.image(str(best_plot), use_column_width=True)

        st.markdown("---")
        st.subheader("🔍 Importance des features")
        fi_plot = PROJECT_ROOT / "plots" / "04_feature_importance.png"
        if fi_plot.exists():
            st.image(str(fi_plot), use_column_width=True)
        st.info(
            "Les features **Menstrual_Irregularity** et **Chronic_Pain_Level** "
            "dominent largement la prédiction, ce qui est cohérent avec la clinique."
        )

    # ── Section 3 : Démo interactive ─────────────────────────
    elif section == "🤖 Démo interactive":
        st.title("🤖 Démo — Prédiction en temps réel")
        st.markdown(
            """
            Renseignez les caractéristiques cliniques de la patiente ci-dessous.
            Le modèle **Gradient Boosting** (le plus performant) prédit le risque d'endométriose.
            """
        )

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Caractéristiques cliniques")
            age = st.slider("Âge", min_value=18, max_value=50, value=30, step=1)
            chronic_pain = st.slider("Niveau de douleur chronique (0–10)", 0.0, 10.0, 5.0, 0.5)
            bmi = st.slider("IMC (BMI)", 15.0, 40.0, 24.0, 0.5)

        with col2:
            st.subheader("Symptômes binaires")
            menstrual_irreg = st.checkbox("Irrégularités menstruelles")
            hormone_abnorm = st.checkbox("Anomalie hormonale")
            infertility = st.checkbox("Infertilité")

        # Feature engineering (même pipeline que l'entraînement)
        high_pain = int(chronic_pain > 5)
        risk_score = int(menstrual_irreg) + int(hormone_abnorm) + int(infertility)

        # Age_Group one-hot
        if age <= 25:
            age_young, age_mid, age_senior = 1, 0, 0
        elif age <= 35:
            age_young, age_mid, age_senior = 0, 1, 0
        else:
            age_young, age_mid, age_senior = 0, 0, 1

        # StandardScaling (paramètres appris sur le train — valeurs approximées)
        # On utilise le modèle directement avec les features dans leur état original
        # puis on scale manuellement en utilisant les stats du train
        # Pour la démo, on passe les features déjà engineerées (non scalées pour les binaires)
        # Note : les features numériques sont scalées dans load_dataset_split
        # Pour la démo on reconstruit avec le scaler (simplifié : on laisse le modèle gérer)

        features = pd.DataFrame([{
            "Age": age,
            "Menstrual_Irregularity": int(menstrual_irreg),
            "Chronic_Pain_Level": chronic_pain,
            "Hormone_Level_Abnormality": int(hormone_abnorm),
            "Infertility": int(infertility),
            "BMI": bmi,
            "High_Pain": high_pain,
            "Risk_Score": risk_score,
            "Age_young": age_young,
            "Age_mid": age_mid,
            "Age_senior": age_senior,
        }])

        st.markdown("---")

        if st.button("🔍 Prédire", type="primary", use_container_width=True):
            try:
                model = load_model("gradient_boosting")
                scaler, numeric_cols = load_scaler()

                # Scaling avec le vrai scaler fitté sur le train set
                features_scaled = features.copy()
                if scaler is not None and numeric_cols:
                    features_scaled[numeric_cols] = scaler.transform(
                        features_scaled[numeric_cols]
                    )

                pred = model.predict(features_scaled)[0]
                proba = model.predict_proba(features_scaled)[0]

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    if pred == 1:
                        st.error("⚠️ **Risque élevé d'endométriose détecté**")
                    else:
                        st.success("✅ **Faible risque d'endométriose**")

                with col_res2:
                    st.metric("Probabilité endométriose", f"{proba[1]*100:.1f} %")
                    st.metric("Probabilité absence", f"{proba[0]*100:.1f} %")

                # Jauge de risque
                st.markdown("#### Niveau de risque")
                risk_pct = proba[1]
                color = "#fc8d62" if risk_pct > 0.5 else "#66c2a5"
                st.progress(float(risk_pct))

                # Facteurs clés
                st.markdown("#### Facteurs de risque identifiés")
                factors = []
                if menstrual_irreg:
                    factors.append("✗ Irrégularités menstruelles (corrélation 0.70 avec la target)")
                if chronic_pain > 5:
                    factors.append(f"✗ Douleur sévère (niveau {chronic_pain}/10)")
                if hormone_abnorm:
                    factors.append("✗ Anomalie hormonale")
                if infertility:
                    factors.append("✗ Infertilité")
                if risk_score >= 2:
                    factors.append(f"✗ Risk Score élevé ({risk_score}/3)")

                if factors:
                    for f in factors:
                        st.markdown(f"- {f}")
                else:
                    st.markdown("- Aucun facteur de risque majeur identifié")

                st.caption(
                    "⚠️ Cet outil est un aide à la décision et ne remplace pas un diagnostic médical."
                )

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.info("Assurez-vous d'avoir lancé `python scripts/session4_models.py` d'abord.")


if __name__ == "__main__":
    build_app()

# Assignment 1

**Étudiant** : Noa Attia (nattia-ai)
**Date** : 30/04/2026

---

## Mon projet

**Prédiction de l'endométriose à partir de symptômes cliniques**

L'endométriose est une maladie chronique touchant environ 10% des femmes en âge de procréer, avec un délai de diagnostic moyen de 7 à 10 ans. L'objectif de ce projet est de construire un modèle de machine learning capable de prédire si une patiente est atteinte d'endométriose à partir de ses symptômes et caractéristiques cliniques.

## Le business case

Réduire le délai de diagnostic en proposant un outil d'aide à la décision médicale. Un modèle prédictif performant pourrait permettre aux médecins d'identifier plus rapidement les patientes à risque et d'orienter les examens complémentaires.

**Type de problème** : Classification binaire (0 = pas d'endométriose, 1 = endométriose présente)

## Les sources de données

- **Dataset** : [Endometriosis Dataset](https://www.kaggle.com/datasets/michaelanietie/endometriosis-dataset) (Kaggle)
- **Fichier** : `data/structured_endometriosis_data.csv`
- **Format** : CSV synthétique, 10 000 lignes × 7 colonnes
- **Licence** : MIT

### Features

| Feature | Type | Description |
|---|---|---|
| Age | float | Âge de la patiente (18–50 ans) |
| Menstrual_Irregularity | binary | Irrégularités menstruelles (0/1) |
| Chronic_Pain_Level | float | Niveau de douleur chronique (0–10) |
| Hormone_Level_Abnormality | binary | Anomalie hormonale (0/1) |
| Infertility | binary | Infertilité (0/1) |
| BMI | float | Indice de masse corporelle (15–40) |
| **Diagnosis** | **binary** | **Variable cible** (0 = non, 1 = oui) |

---

## Session 2 — Qualité du dataset

- ✅ Aucune valeur manquante
- ✅ Aucun outlier sur les features (méthode IQR)
- ✅ Aucun feature drift (variation < 5% entre les 5 splits)
- ⚠️ Class imbalance : 75% positifs / 25% négatifs (ratio 3.1x) → `class_weight='balanced'`
- 🔗 Features les plus corrélées à la target : `Menstrual_Irregularity` (0.70), `Chronic_Pain_Level` (0.51)

---

## Session 3 — Feature Engineering & EDA visuelle

### Nouvelles features créées

| Feature | Description | Justification |
|---|---|---|
| `High_Pain` | 1 si douleur > 5/10 | Seuil clinique — douleur sévère |
| `Risk_Score` | Somme des 3 facteurs binaires (0–3) | Score cumulé de risque |
| `Age_young/mid/senior` | Tranche d'âge one-hot | Non-linéarité de l'âge |

### Pipeline de données

1. Feature engineering
2. Train/test split **80/20 stratifié** (préserve le ratio 75/25)
3. **StandardScaler** fitté sur train uniquement (pas de data leakage)
4. **PCA** : 11 features → 7 composantes (97.9% variance conservée)
5. **skrub TableVectorizer** : encodage automatique de toutes les colonnes

### Visualisations (`deliverables/plots/`)

- `01_distributions.png` — histogrammes de toutes les features
- `02_features_by_target.png` — distributions par classe
- `03_correlation_heatmap.png` — matrice de corrélation
- `04_class_imbalance.png` — déséquilibre de classes
- `05_new_features.png` — Risk Score et High Pain vs Diagnosis

---

## Session 4 — Modèles ML

### Modèles entraînés

| Modèle | Type | Paramètres clés |
|---|---|---|
| Logistic Regression | Baseline linéaire | `max_iter=1000`, `class_weight=balanced` |
| Random Forest | Ensemble (bagging) | `n_estimators=200`, `max_depth=10` |
| **Gradient Boosting** ✅ | Ensemble (boosting) | `n_estimators=200`, `lr=0.05`, `max_depth=4` |

### Résultats sur le test set (20%)

| Modèle | Accuracy | Precision | Recall | **F1** | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8990 | 0.9658 | 0.8979 | 0.9306 | 0.9002 |
| Random Forest | 0.9045 | 0.9595 | 0.9118 | 0.9351 | 0.8970 |
| **Gradient Boosting** | **0.9220** | **0.9395** | **0.9582** | **0.9488** | 0.8846 |

**Métrique principale : F1-score** (robuste au déséquilibre de classes)  
**Meilleur modèle : Gradient Boosting** — F1 = 0.9488, validé par cross-validation 5 folds

---

## Session 5 — Dashboard & Visualisations

### Plots GitHub (`plots/`)

- `01_eda_correlation.png` — heatmap de corrélation des features
- `02_model_comparison.png` — comparaison F1 / Accuracy / Recall des 3 modèles
- `03_best_model_results.png` — confusion matrix + courbe ROC (Gradient Boosting)
- `04_feature_importance.png` — importance des features (Gradient Boosting)

### Dashboard Streamlit (`src/app.py`)

3 sections :
1. **Problème & EDA** — contexte médical, aperçu du dataset, visualisations exploratoires
2. **Comparaison des modèles** — tableau de métriques, graphiques, confusion matrix, ROC, feature importance
3. **Démo interactive** — prédiction en temps réel à partir des symptômes d'une patiente

Lancer avec : `python scripts/main.py`

# Prédiction de l'endométriose — ML Proof of Concept

**Étudiant :** Noa Attia ([@nattia-ai](https://github.com/nattia-ai))  
**Cours :** ML Proof of Concept  
**Dataset :** [Endometriosis Dataset — Kaggle](https://www.kaggle.com/datasets/michaelanietie/endometriosis-dataset)

---

## Problème métier

L'endométriose touche ~10 % des femmes en âge de procréer avec un délai de diagnostic moyen de **7 à 10 ans**. Ce projet construit un modèle de classification binaire pour prédire si une patiente est atteinte d'endométriose à partir de ses symptômes cliniques, afin d'aider les médecins à identifier plus rapidement les patientes à risque.

**Type de problème :** Classification binaire  
**Target :** `Diagnosis` (0 = non, 1 = endométriose)  
**Dataset :** 10 000 patientes × 7 features cliniques

---

## Résultats

| Modèle | Accuracy | F1-score | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.8990 | 0.9306 | 0.9002 |
| Random Forest | 0.9045 | 0.9351 | 0.8970 |
| **Gradient Boosting** ✅ | **0.9220** | **0.9488** | **0.8846** |

**Meilleur modèle :** Gradient Boosting (F1 = 0.9488)

---

## Structure du projet

```
ml-poc-project/
├── data/                   # Dataset CSV
├── deliverables/           # Rendus (assignments + plots EDA)
├── models/                 # Modèles entraînés (.pkl) + scaler
├── plots/                  # Visualisations GitHub (4 plots)
├── results/                # model_metrics.csv
├── scripts/
│   ├── main.py             # Point d'entrée principal
│   ├── session2_eda.py     # Checks qualité du dataset
│   ├── session3_eda.py     # Visualisations EDA
│   ├── session4_models.py  # Entraînement des 3 modèles
│   └── session5_plots.py   # Génération des plots GitHub
└── src/
    ├── app.py              # Dashboard Streamlit
    ├── config.py           # Configuration et registre des modèles
    ├── data.py             # Chargement + pipeline de données
    ├── features.py         # Feature engineering + encodage
    ├── metrics.py          # Métriques d'évaluation
    └── model_io.py         # Chargement des modèles
```

---

## Lancer le projet

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Entraîner les modèles

```bash
python scripts/session4_models.py
```

Cela entraîne les 3 modèles ML et sauvegarde les `.pkl` dans `models/`.

### 3. Générer les plots

```bash
python scripts/session5_plots.py
```

### 4. Lancer le dashboard complet

```bash
python scripts/main.py
```

Ouvre ensuite [http://localhost:8501](http://localhost:8501) dans le navigateur.

Le dashboard Streamlit contient 3 sections :
- **Problème & EDA** — contexte médical, aperçu du dataset, visualisations
- **Comparaison des modèles** — tableau de métriques, graphiques, confusion matrix, courbe ROC
- **Démo interactive** — prédiction en temps réel à partir des symptômes d'une patiente

---

## Pipeline de données

1. Chargement du CSV (10 000 lignes)
2. Feature engineering : `High_Pain`, `Risk_Score`, `Age_Group` (one-hot)
3. Train/test split 80/20 **stratifié** (préserve le ratio 75/25)
4. StandardScaling des features numériques (fitté sur train uniquement — pas de data leakage)

---

## Choix techniques

- **class_weight='balanced'** sur tous les modèles pour compenser le déséquilibre de classes (75/25)
- **F1-score** comme métrique principale (robuste à l'imbalance)
- **Cross-validation 5 folds** pour valider la stabilité des performances
- Scaler sauvegardé (`models/scaler.pkl`) pour cohérence entre entraînement et démo

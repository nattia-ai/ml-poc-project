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

### Observations principales (Session 2)

- ✅ Aucune valeur manquante
- ✅ Aucun outlier sur les features
- ⚠️ Class imbalance : 75% positifs / 25% négatifs (ratio 3.1x) → prévoir `class_weight='balanced'`
- 🔗 Features les plus corrélées à la target : `Menstrual_Irregularity` (0.70), `Chronic_Pain_Level` (0.51)

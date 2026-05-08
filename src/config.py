from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Session 4 — Modèles entraînés (Prédiction de l'endométriose)
MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression (baseline)",
        "description": "Modèle linéaire de référence avec class_weight='balanced'.",
        "path": MODELS_DIR / "logistic_regression.pkl",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble de 200 arbres de décision (bagging).",
        "path": MODELS_DIR / "random_forest.pkl",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "description": "Boosting séquentiel — généralement le plus performant.",
        "path": MODELS_DIR / "gradient_boosting.pkl",
    },
}

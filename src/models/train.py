"""
Entraînement modèle de détection d'anomalies (Isolation Forest).
- charge données prétraitées (ou brutes si nécessaire)
- entraîne le modèle
- sauvegarde le modèle avec joblib
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest


def train_model(
    input_csv: str = "data/processed/sensor_stream_preprocessed.csv",
    model_path: str = "data/processed/model.joblib",
    contamination: float = 0.03,
    random_state: int = 42,
) -> None:
    """
    Entraîne un modèle IsolationForest sur les features sélectionnées.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset prétraité introuvable: {input_csv}. "
            f"Exécute d'abord src/preprocessing/preprocess.py"
        )

    df = pd.read_csv(input_path)

    # Features minimales pour baseline
    required = {"value", "delta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Features manquantes pour entraînement: {missing}")

    X = df[["value", "delta"]]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)

    print(f"[OK] Modèle entraîné et sauvegardé: {model_file}")
    print(f"Nombre d'échantillons: {len(df)}")


if __name__ == "__main__":
    train_model()
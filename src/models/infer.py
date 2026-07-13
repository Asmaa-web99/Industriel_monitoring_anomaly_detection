"""
Inférence du modèle d'anomalies.
Convention IsolationForest:
- predict() retourne  1 => normal
- predict() retourne -1 => anomalie
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import joblib


DEFAULT_MODEL_PATH = "data/processed/model.joblib"


def predict_one(value: float, delta: float = 0.0, model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """
    Prédit l'état d'un point capteur unique.
    Retour:
      - is_anomaly (bool)
      - raw_prediction (int)
      - anomaly_score (float) : score converti [0..1] approx
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {model_path}. Lance d'abord src/models/train.py"
        )

    model = joblib.load(model_file)

    # X shape = (1, n_features)
    X = np.array([[value, delta]], dtype=float)

    raw_pred = int(model.predict(X)[0])  # 1 normal / -1 anomaly
    # decision_function > 0 normal, < 0 anomalie
    decision = float(model.decision_function(X)[0])

    # Transformation simple en score [0..1] pour API/dash
    # plus decision est négative, plus c'est suspect
    anomaly_score = 1.0 / (1.0 + np.exp(5 * decision))  # sigmoid inversée approx

    return {
        "is_anomaly": raw_pred == -1,
        "raw_prediction": raw_pred,
        "anomaly_score": round(float(anomaly_score), 4),
        "decision_function": round(decision, 6),
    }
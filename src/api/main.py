"""
API FastAPI pour monitoring industriel:
- /health : état service
- /detect : détection d'anomalie point par point
- /detect/batch : détection sur une liste de points
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.infer import predict_one


app = FastAPI(
    title="Industrial Monitoring API",
    version="0.1.0",
    description="API de détection d'anomalies pour flux IoT industriels",
)


class SensorPoint(BaseModel):
    """
    Schéma entrée capteur unitaire.
    delta est optionnel (variation), défaut = 0.0
    """
    timestamp: str = Field(..., example="2026-07-13T10:00:00Z")
    value: float = Field(..., example=52.4)
    delta: Optional[float] = Field(0.0, example=0.7)


class DetectResponse(BaseModel):
    timestamp: str
    value: float
    delta: float
    is_anomaly: bool
    anomaly_score: float
    raw_prediction: int
    decision_function: float


@app.get("/health")
def health():
    """
    Endpoint de santé pour vérifier que l'API tourne.
    """
    return {"status": "ok", "service": "industrial-monitoring-api"}


@app.post("/detect", response_model=DetectResponse)
def detect(point: SensorPoint):
    """
    Détection d'anomalie pour un point.
    """
    try:
        result = predict_one(value=point.value, delta=point.delta or 0.0)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur détection: {e}")

    return DetectResponse(
        timestamp=point.timestamp,
        value=point.value,
        delta=point.delta or 0.0,
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        raw_prediction=result["raw_prediction"],
        decision_function=result["decision_function"],
    )


@app.post("/detect/batch")
def detect_batch(points: List[SensorPoint]):
    """
    Détection d'anomalie sur une liste de points (utile pour dashboard/tests).
    """
    if len(points) == 0:
        raise HTTPException(status_code=400, detail="La liste des points est vide.")

    results = []
    for p in points:
        r = predict_one(value=p.value, delta=p.delta or 0.0)
        results.append(
            {
                "timestamp": p.timestamp,
                "value": p.value,
                "delta": p.delta or 0.0,
                **r,
            }
        )

    n_anomalies = sum(1 for x in results if x["is_anomaly"])
    return {
        "count": len(results),
        "anomalies": n_anomalies,
        "results": results,
    }
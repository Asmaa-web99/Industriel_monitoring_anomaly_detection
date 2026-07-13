
# Industriel Monitoring Anomaly Detection

Prototype de monitoring industriel intelligent pour la détection d’anomalies et l’alerte temps réel à partir de données IoT.

## Stack

- Python (FastAPI, scikit-learn, pandas)
- Détection: Isolation Forest / One-Class SVM / Autoencoder (optionnel au début)
- Visualisation: Plotly (Grafana plus tard)
- Conteneurisation: Docker / Docker Compose

## Structure

Voir `docs/architecture.md`.

## Lancement rapide

```bash
cp .env.example .env
docker compose up --build
```

"""
Simulation de flux capteurs industriels (série temporelle) avec injection d'anomalies.
- Génère un signal "normal" (sinusoïde + bruit)
- Injecte des anomalies (pics brutaux)
- Sauvegarde en CSV pour la suite du pipeline
"""

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_sensor_data(
    n_points: int = 1000,
    base_value: float = 50.0,
    noise_std: float = 1.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Génère une série temporelle simulée:
    - timestamp: cadence de 1 seconde
    - value: signal de capteur
    - is_anomaly: label binaire (0 normal, 1 anomalie)
    """
    rng = np.random.default_rng(seed)

    # Axe temps
    start_time = datetime.now(timezone.utc).replace(microsecond=0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_points)]

    # Signal normal: composante sinusoïdale + faible tendance + bruit gaussien
    t = np.linspace(0, 30, n_points)
    seasonal = 5 * np.sin(t)                  # variation périodique
    trend = np.linspace(0, 1.5, n_points)     # petite dérive réaliste
    noise = rng.normal(0, noise_std, n_points)

    values = base_value + seasonal + trend + noise
    labels = np.zeros(n_points, dtype=int)    # 0 = normal

    # Injection d'anomalies (pics) à des positions fixes
    anomaly_ranges = [(250, 260), (600, 608), (850, 860)]
    for start, end in anomaly_ranges:
        if start < n_points:
            end = min(end, n_points)
            # pic positif marqué
            values[start:end] += rng.uniform(12, 20)
            labels[start:end] = 1

    # Créer DataFrame final
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": values.round(4),
            "is_anomaly": labels,
        }
    )
    return df


def main() -> None:
    """
    Point d'entrée script:
    - Crée le dossier data/sample si besoin
    - Génère le CSV de simulation
    """
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_sensor_data()
    output_path = output_dir / "sensor_stream.csv"
    df.to_csv(output_path, index=False)

    print(f"[OK] Dataset simulé généré: {output_path}")
    print(df.head(5))
    print(df.tail(5))
    print(f"Total points: {len(df)} | Anomalies: {int(df['is_anomaly'].sum())}")


if __name__ == "__main__":
    main()
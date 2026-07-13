"""
Prétraitement des séries temporelles:
- validation colonnes
- gestion des valeurs manquantes
- conversion timestamp
- normalisation de la variable capteur
- création de features temporelles simples
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = {"timestamp", "value"}


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Prétraite un DataFrame capteur.
    Retourne:
      - DataFrame enrichi
      - scaler entraîné (pour réutilisation en inférence)
    """
    # Vérifier présence des colonnes minimales
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    out = df.copy()

    # Convertir timestamp en datetime
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # Supprimer lignes invalides (timestamp ou value manquants)
    out = out.dropna(subset=["timestamp", "value"]).reset_index(drop=True)

    # Feature de base: variation absolue entre points successifs
    out["delta"] = out["value"].diff().fillna(0.0)

    # Normaliser la colonne principale "value"
    scaler = StandardScaler()
    out["value_scaled"] = scaler.fit_transform(out[["value"]])

    # Normaliser la feature delta (optionnel mais utile pour modèles)
    delta_scaler = StandardScaler()
    out["delta_scaled"] = delta_scaler.fit_transform(out[["delta"]])

    return out, scaler


def run_preprocessing(
    input_csv: str = "data/sample/sensor_stream.csv",
    output_csv: str = "data/processed/sensor_stream_preprocessed.csv",
) -> None:
    """
    Exécute le prétraitement complet et sauvegarde le résultat.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_csv}")

    df = pd.read_csv(input_path)
    preprocessed_df, _ = preprocess(df)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessed_df.to_csv(output_path, index=False)

    print(f"[OK] Prétraitement terminé: {output_path}")
    print(preprocessed_df.head(5))


if __name__ == "__main__":
    run_preprocessing()
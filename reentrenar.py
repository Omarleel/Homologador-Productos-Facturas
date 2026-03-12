from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import construir_dataset_entrenamiento, init_seeds
from utils.config import (
    dataset_path,
    ensure_project_dirs,
    model_path,
    require_file,
)
from core.model import ModeloMatchProducto


MODELO_NOMBRE = "homologacion_v2"

ARCHIVO_MAESTRO = "maestro.csv"
ARCHIVO_HISTORIAL_INCREMENTAL = "historial_facturas_incremental.csv"
ARCHIVO_PARES_REENTRENAMIENTO = "pares_reentrenamiento_incremental.csv"


def crear_backup_modelo_si_existe(modelo_dir: Path) -> None:
    if not modelo_dir.exists():
        raise FileNotFoundError(
            f"No existe el modelo base para reentrenar: '{modelo_dir}'"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_backup = modelo_dir.parent / f"{modelo_dir.name}_backup_{timestamp}"

    shutil.copytree(modelo_dir, ruta_backup)
    print(f"Backup del modelo anterior creado en: {ruta_backup}")


def main() -> None:
    init_seeds()
    ensure_project_dirs()

    maestro_path = require_file(dataset_path(ARCHIVO_MAESTRO), "dataset maestro")
    incremental_path = require_file(
        dataset_path(ARCHIVO_HISTORIAL_INCREMENTAL),
        "dataset historial_facturas_incremental",
    )
    ruta_modelo = model_path(MODELO_NOMBRE)

    maestro = pd.read_csv(maestro_path, encoding="utf-8-sig")
    historial_incremental = pd.read_csv(incremental_path, encoding="utf-8-sig")

    print("COLUMNAS maestro:", [repr(c) for c in maestro.columns])
    print(
        "COLUMNAS historial incremental:",
        [repr(c) for c in historial_incremental.columns],
    )

    pares = construir_dataset_entrenamiento(
        maestro=maestro,
        historial_facturas=historial_incremental,
        n_neg_por_pos=5,
    )

    print("Pares de reentrenamiento incremental:", pares.shape)
    print(pares["label"].value_counts(dropna=False))

    pares_path = dataset_path(ARCHIVO_PARES_REENTRENAMIENTO)
    pares.to_csv(pares_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Dataset incremental de pares guardado en: {pares_path}")

    crear_backup_modelo_si_existe(ruta_modelo)

    modelo = ModeloMatchProducto.cargar(ruta_modelo)

    print("\nModelo cargado para reentrenamiento incremental")
    print(f"best_threshold anterior: {modelo.best_threshold:.4f}")
    print(f"max_tokens: {modelo.config.max_tokens}")
    print(f"text_embedding_dim: {modelo.config.text_embedding_dim}")
    print(f"learning_rate: {modelo.config.learning_rate}")

    modelo.fit_incremental(
        pares=pares,
        epochs=6,
        batch_size=256,
        recalcular_threshold=True,
    )

    modelo.guardar(ruta_modelo)

    print("\n--- REENTRENAMIENTO INCREMENTAL FINALIZADO ---")
    print(f"Modelo actualizado en: '{ruta_modelo}'")
    print(f"Nuevo best_threshold: {modelo.best_threshold:.4f}")


if __name__ == "__main__":
    main()
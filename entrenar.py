from __future__ import annotations

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
ARCHIVO_HISTORIAL_FACTURAS = "historial_facturas.csv"
ARCHIVO_PARES_ENTRENAMIENTO = "pares_entrenamiento.csv"


def main() -> None:
    init_seeds()
    ensure_project_dirs()

    maestro_path = require_file(dataset_path(ARCHIVO_MAESTRO), "dataset maestro")
    historial_path = require_file(
        dataset_path(ARCHIVO_HISTORIAL_FACTURAS),
        "dataset historial_facturas",
    )

    maestro = pd.read_csv(maestro_path, encoding="utf-8-sig")
    historial = pd.read_csv(historial_path, encoding="utf-8-sig")

    print("COLUMNAS maestro:", [repr(c) for c in maestro.columns])
    print("COLUMNAS historial:", [repr(c) for c in historial.columns])

    pares = construir_dataset_entrenamiento(
        maestro=maestro,
        historial_facturas=historial,
        n_neg_por_pos=5,
    )

    print("Pares de entrenamiento:", pares.shape)
    print(pares["label"].value_counts(dropna=False))

    pares_path = dataset_path(ARCHIVO_PARES_ENTRENAMIENTO)
    pares.to_csv(pares_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"Dataset de pares guardado en: {pares_path}")

    modelo = ModeloMatchProducto(
        max_tokens=12000,
        text_embedding_dim=32,
    )

    modelo.fit(
        pares=pares,
        epochs=16,
        batch_size=256,
    )

    ruta_modelo = model_path(MODELO_NOMBRE)
    modelo.guardar(ruta_modelo)

    print("\n--- ENTRENAMIENTO FINALIZADO ---")
    print(f"Modelo guardado en: '{ruta_modelo}'")


if __name__ == "__main__":
    main()
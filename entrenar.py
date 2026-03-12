import pandas as pd

from homologacion import (
    init_seeds,
    construir_dataset_entrenamiento,
)

from homologacion.modelo import ModeloMatchCodProducto

def main() -> None:
    init_seeds()

    maestro = pd.read_csv("maestro.csv", encoding="utf-8-sig")
    historial = pd.read_csv("historial_facturas.csv", encoding="utf-8-sig")

    print("COLUMNAS maestro:", [repr(c) for c in maestro.columns])
    print("COLUMNAS historial:", [repr(c) for c in historial.columns])

    pares = construir_dataset_entrenamiento(
        maestro=maestro,
        historial_facturas=historial,
        n_neg_por_pos=4,
    )

    print("Pares de entrenamiento:", pares.shape)
    print(pares["label"].value_counts(dropna=False))

    modelo = ModeloMatchCodProducto(
        max_tokens=6000,
        text_embedding_dim=24,
    )

    modelo.fit(
        pares=pares,
        epochs=12,
        batch_size=128,
    )

    modelo.guardar("modelo_homologacion_v1")


if __name__ == "__main__":
    main()
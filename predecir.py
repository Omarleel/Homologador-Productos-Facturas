import pandas as pd

from homologacion import (
    init_seeds,
    inferir_codproducto,
    ModeloMatchCodProducto,
)


def main() -> None:
    init_seeds()

    maestro = pd.read_csv("maestro.csv", encoding="utf-8-sig")
    nuevas = pd.read_csv("facturas_nuevas.csv", encoding="utf-8-sig")

    print("COLUMNAS maestro:", [repr(c) for c in maestro.columns])
    print("COLUMNAS nuevas:", [repr(c) for c in nuevas.columns])

    modelo = ModeloMatchCodProducto.cargar("modelo_homologacion_v1")

    resultado = inferir_codproducto(
        facturas_nuevas=nuevas,
        maestro=maestro,
        modelo_match=modelo,
        top_k=3,
        umbral_match=modelo.best_threshold,
        top_n_candidatos=30,
    )

    resultado.to_csv("resultado_inferencia.csv", index=False, encoding="utf-8-sig")
    print(resultado.head(20))


if __name__ == "__main__":
    main()
from __future__ import annotations

import pandas as pd

from utils import inferir_codproducto, init_seeds
from utils.config import (
    dataset_path,
    ensure_project_dirs,
    model_path,
    require_file,
    result_path,
)
from core.model import ModeloMatchProducto


MODELO_NOMBRE = "homologacion_v2"
ARCHIVO_MAESTRO = "maestro.csv"
ARCHIVO_FACTURAS_NUEVAS = "facturas_nuevas_v2.csv"

ARCHIVO_RESULTADO_TECNICO = "resultado_inferencia.csv"
ARCHIVO_RESULTADO_RESUMIDO = "match_final_resumido.csv"


def main() -> None:
    init_seeds()
    ensure_project_dirs()

    maestro_path = require_file(dataset_path(ARCHIVO_MAESTRO), "dataset maestro")
    nuevas_path = require_file(
        dataset_path(ARCHIVO_FACTURAS_NUEVAS),
        "dataset facturas_nuevas",
    )
    ruta_modelo = require_file(model_path(MODELO_NOMBRE), "directorio del modelo")

    maestro = pd.read_csv(maestro_path, encoding="utf-8-sig")
    nuevas = pd.read_csv(nuevas_path, encoding="utf-8-sig")

    print("COLUMNAS maestro:", [repr(c) for c in maestro.columns])
    print("COLUMNAS nuevas:", [repr(c) for c in nuevas.columns])

    modelo = ModeloMatchProducto.cargar(ruta_modelo)

    resultado = inferir_codproducto(
        facturas_nuevas=nuevas,
        maestro=maestro,
        modelo_match=modelo,
        top_k=5,
        umbral_match=modelo.best_threshold,
        top_n_candidatos=40,
    )

    resultado_path = result_path(ARCHIVO_RESULTADO_TECNICO)
    resultado.to_csv(resultado_path, sep=";", index=False, encoding="utf-8-sig")

    resumido = resultado[resultado["Rank"] == 1].copy()

    columnas_finales = [
        "RucProveedor",
        "CodFactura",
        "ProductoFactura",
        "UnidadFactura",
        "CodProducto",
        "Producto",
        "TipoResultado",
    ]
    resumido = resumido[columnas_finales]

    resumido.columns = [
        "RUC_PROVEEDOR",
        "COD_PRODUCTO_FACTURA",
        "NOMBRE_PRODUCTO_FACTURA",
        "UM_FACTURA",
        "COD_PRODUCTO_SISTEMA",
        "NOMBRE_PRODUCTO_SISTEMA",
        "ESTADO_MATCH",
    ]

    resumido_path = result_path(ARCHIVO_RESULTADO_RESUMIDO)
    resumido.to_csv(resumido_path, sep=";", index=False, encoding="utf-8-sig")

    print("\n--- PROCESO FINALIZADO ---")
    print(f"Archivo técnico: '{resultado_path}'")
    print(f"Archivo resumen: '{resumido_path}'")
    print(resumido.head(10))


if __name__ == "__main__":
    main()
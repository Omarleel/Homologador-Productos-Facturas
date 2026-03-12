import pandas as pd

from homologacion import (
    inferir_codproducto,
    init_seeds,
)
from homologacion.modelo_2 import ModeloMatchCodProducto

def main() -> None:
    init_seeds()

    maestro = pd.read_csv("maestro.csv", encoding="utf-8-sig")
    nuevas = pd.read_csv("facturas_nuevas.csv", encoding="utf-8-sig")

    print("COLUMNAS maestro:", [repr(c) for c in maestro.columns])
    print("COLUMNAS nuevas:", [repr(c) for c in nuevas.columns])

    modelo = ModeloMatchCodProducto.cargar("modelo_homologacion_v2")

    resultado = inferir_codproducto(
        facturas_nuevas=nuevas,
        maestro=maestro,
        modelo_match=modelo,
        top_k=3,
        umbral_match=modelo.best_threshold,
        top_n_candidatos=40,
    )

    resultado.to_csv("resultado_inferencia.csv", sep=';', index=False, encoding="utf-8-sig")
    
    resumido = resultado[resultado["Rank"] == 1].copy()

    # Seleccionamos solo las columnas clave para el usuario final
    columnas_finales = [
        "RucProveedor",
        "CodFactura",
        "ProductoFactura",
        "UnidadFactura",
        "CodProducto",       # Este es el código en tu Maestro
        "Producto",          # Este es el nombre en tu Maestro
        "TipoResultado"      # Indica si fue EXACTO, TENTATIVO o POSIBLE_NUEVO
    ]

    # Nos aseguramos de que solo existan estas columnas
    resumido = resumido[columnas_finales]

    # Renombramos para que sea más claro para quien lea el Excel/CSV
    resumido.columns = [
        "RUC_PROVEEDOR",
        "COD_PRODUCTO_FACTURA",
        "NOMBRE_PRODUCTO_FACTURA",
        "UM_FACTURA",
        "COD_PRODUCTO_SISTEMA",
        "NOMBRE_PRODUCTO_SISTEMA",
        "ESTADO_MATCH"
    ]

    resumido.to_csv("match_final_resumido.csv", sep=';', index=False, encoding="utf-8-sig")
    
    print("\n--- PROCESO FINALIZADO ---")
    print("Archivo técnico: 'resultado_inferencia_completo.csv'")
    print("Archivo resumen: 'match_final_resumido.csv'")
    print(resumido.head(10))


if __name__ == "__main__":
    main()
import pandas as pd

from .limpieza import (
    renombrar_columnas_equivalentes,
    validar_columnas,
    normalizar_codigo,
    normalizar_texto,
    normalizar_unidad,
    log_seguro,
    extraer_peso_desde_texto
)


EQUIVALENCIAS_MAESTRO = {
    "RucProveedor": ["rucproveedor", "ruc_proveedor", "ruc"],
    "CodProducto": ["codigo_producto", "codigoproducto", "codigo", "codproducto"],
    "CodProducto2": ["codigo_producto2", "codigoproducto2", "codproducto2"],
    "CodProducto3": ["codigo_producto3", "codigoproducto3", "codproducto3"],
    "Producto": ["producto", "descripcion", "descripción"],
    "UnidaMedidaCompra": ["unidadmedidacompra", "unidamedidacompra", "unidad_compra", "unidad"],
    "CostoCaja": ["costocaja", "costo_caja", "costo"],
}

EQUIVALENCIAS_FACTURAS = {
    "RucProveedor": ["rucproveedor", "ruc_proveedor", "ruc"],
    "CodProducto": ["codigo_producto", "codigoproducto", "codigo", "codproducto"],
    "Producto": ["producto", "descripcion", "descripción"],
    "UnidaMedidaCompra": [
        "unidadmedidacompra",
        "unidamedidacompra",
        "unidad_compra",
        "unidad",
        "unidadmedida",
    ],
    "CostoCaja": ["costocaja", "costo_caja", "costo"],
}


def preparar_maestro(maestro: pd.DataFrame) -> pd.DataFrame:
    df = renombrar_columnas_equivalentes(maestro, EQUIVALENCIAS_MAESTRO)

    validar_columnas(
        df,
        ["RucProveedor", "CodProducto", "Producto", "UnidaMedidaCompra"],
        "maestro",
    )

    for c in ["CodProducto", "CodProducto2", "CodProducto3"]:
        if c not in df.columns:
            df[c] = ""

        df[c] = (
            df[c]
            .fillna("")
            .astype(str)
            .replace("nan", "")
            .map(normalizar_codigo)
        )

    if "CostoCaja" not in df.columns:
        df["CostoCaja"] = 0.0

    df["Producto_norm"] = df["Producto"].fillna("").astype(str).map(normalizar_texto)
    if "PesoUnitario" not in df.columns:
        df["PesoUnitario"] = 0.0
    df["PesoUnitario"] = pd.to_numeric(df["PesoUnitario"], errors="coerce").fillna(0.0)
    df["Unidad_norm"] = df["UnidaMedidaCompra"].fillna("").astype(str).map(normalizar_unidad)
    df["CostoCaja"] = pd.to_numeric(df["CostoCaja"], errors="coerce").fillna(0.0)
    df["Costo_log"] = df["CostoCaja"].map(log_seguro)

    return df


def preparar_facturas(facturas: pd.DataFrame) -> pd.DataFrame:
    df = renombrar_columnas_equivalentes(facturas, EQUIVALENCIAS_FACTURAS)

    validar_columnas(
        df,
        ["RucProveedor", "CodProducto", "Producto", "UnidaMedidaCompra"],
        "facturas",
    )

    if "CostoCaja" not in df.columns:
        df["CostoCaja"] = 0.0

    df["CodProducto"] = (
        df["CodProducto"]
        .fillna("")
        .astype(str)
        .replace("nan", "")
        .map(normalizar_codigo)
    )

    df["Producto_norm"] = df["Producto"].fillna("").astype(str).map(normalizar_texto)
    if "PesoUnitario" not in df.columns:
        # Opcional: intentar extraer del texto
        df["PesoUnitario"] = df["Producto"].apply(extraer_peso_desde_texto)
    else:
        df["PesoUnitario"] = pd.to_numeric(df["PesoUnitario"], errors="coerce").fillna(0.0)
    df["Unidad_norm"] = df["UnidaMedidaCompra"].fillna("").astype(str).map(normalizar_unidad)
    df["CostoCaja"] = pd.to_numeric(df["CostoCaja"], errors="coerce").fillna(0.0)
    df["Costo_log"] = df["CostoCaja"].map(log_seguro)

    return df
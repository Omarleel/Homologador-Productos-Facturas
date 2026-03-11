import re
import unicodedata
import pandas as pd


def limpiar_nombres_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def renombrar_columnas_equivalentes(df: pd.DataFrame, equivalencias: dict) -> pd.DataFrame:
    df = limpiar_nombres_columnas(df)

    columnas_lower = {c.lower(): c for c in df.columns}
    renombres = {}

    for canonica, aliases in equivalencias.items():
        encontrada = None

        for nombre in [canonica] + aliases:
            key = nombre.lower().strip().replace("\ufeff", "")
            if key in columnas_lower:
                encontrada = columnas_lower[key]
                break

        if encontrada is not None and encontrada != canonica:
            renombres[encontrada] = canonica

    if renombres:
        df = df.rename(columns=renombres)

    return df


def validar_columnas(df: pd.DataFrame, requeridas: list, nombre_df: str) -> None:
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"En {nombre_df} faltan columnas requeridas: {faltantes}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


def quitar_acentos(texto: str) -> str:
    if pd.isna(texto):
        return ""
    texto = str(texto)
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def normalizar_texto(texto: str) -> str:
    texto = quitar_acentos(texto).upper().strip()
    texto = re.sub(r"[^A-Z0-9]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def normalizar_codigo(codigo: str) -> str:
    if pd.isna(codigo):
        return ""
    codigo = str(codigo).strip().upper()
    codigo = re.sub(r"\s+", "", codigo)
    return codigo


def normalizar_unidad(u: str) -> str:
    u = normalizar_texto(u)
    mapa = {
        "UNIDAD PIEZA": "UNIDAD",
        "UNIDAD PIEZAS": "UNIDAD",
        "UND": "UNIDAD",
        "UND PIEZA": "UNIDAD",
        "PIEZA": "UNIDAD",
        "CAJ": "CAJA",
        "CJ": "CAJA",
    }
    return mapa.get(u, u)


def log_seguro(x: float) -> float:
    x = 0.0 if pd.isna(x) else float(x)
    return float(__import__("numpy").log1p(max(x, 0.0)))
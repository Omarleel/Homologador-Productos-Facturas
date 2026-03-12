from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .limpieza import normalizar_codigo, normalizar_texto


def construir_indice_codigos(maestro: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    indice: Dict[Tuple[str, str], int] = {}

    for idx, row in maestro.iterrows():
        ruc = str(row["RucProveedor"]).strip()

        for c in ["CodProducto", "CodProducto2", "CodProducto3"]:
            codigo = row.get(c, "")
            if codigo:
                indice[(ruc, codigo)] = idx

    return indice


def buscar_match_exacto(
    fila_factura: pd.Series,
    maestro: pd.DataFrame,
    indice_codigos: Dict[Tuple[str, str], int],
) -> Optional[pd.Series]:
    key = (
        str(fila_factura["RucProveedor"]).strip(),
        normalizar_codigo(fila_factura["CodProducto"]),
    )
    idx = indice_codigos.get(key)

    if idx is None:
        return None

    return maestro.loc[idx]


def token_set(texto: str) -> set:
    return set(normalizar_texto(texto).split())


def jaccard(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / max(len(sa | sb), 1)


def bonus_marca(fact_text: str, master_text: str) -> float:
    claves = claves = [
        # P&G / Cuidado Personal
        "PANTENE", "HEAD", "SHOULDERS", "DOWNY", "ARIEL", "GILLETTE",
        "OLD", "SPICE", "SECRET", "ORAL", "B", "VICK", "PAMPERS",
        "VENUS", "ACE", "PROTEX", "COLGATE", "KOLYNOS", "NINET",
        
        # Alicorp / Limpieza y Alimentos
        "BOLIVAR", "OPAL", "MARSELLA", "PRIMOR", "ALICORP", "TONDERO", 
        "PALMEROLA", "FORTUNA", "POPEYE", "DON", "VITTORIO", "SAYON",
        
        # Gloria / Lácteos
        "GLORIA", "BONLE", "YOFRESH", "PURA", "VIDA",
        
        # Nestlé / Otros
        "NESTLE", "MAGGI", "KIRMA", "NESCAFE", "MILO", "SUBLIME",
        
        # Marcas Locales / Frecuentes en tu dataset
        "FANNY", "POMAROLA", "OSITOS", "ANGEL", "CHOCK", "RICOCAN",
        "THOMAS", "TOTTUS", "GLINA", "CHIN", "GRANUTS", "SCORE", "FOUR", "LOKO"
    ]

    ft = token_set(fact_text)
    mt = token_set(master_text)

    puntos = 0.0
    for k in claves:
        if k in ft and k in mt:
            puntos += 0.05

    return min(puntos, 0.2)


def score_heuristico(fila_factura: pd.Series, fila_maestro: pd.Series) -> float:
    s_texto = jaccard(fila_factura["Producto_norm"], fila_maestro["Producto_norm"])
    s_unidad = 1.0 if fila_factura["Unidad_norm"] == fila_maestro["Unidad_norm"] else 0.0

    # Similitud de Costo
    diff_costo = abs(fila_factura["Costo_log"] - fila_maestro["Costo_log"])
    s_costo = float(np.exp(-2.0 * diff_costo))
    
    # NUEVO: Similitud de Peso (Extraído del texto o columna)
    # Si la factura dice 190g y el maestro 170g, la diferencia es pequeña (0.02kg)
    p_fact = fila_factura.get("PesoUnitario", 0)
    p_maes = fila_maestro.get("PesoUnitario", 0)
    diff_peso = abs(p_fact - p_maes)
    s_peso = float(np.exp(-5.0 * diff_peso)) # Penaliza fuerte diferencias grandes

    score = (0.40 * s_texto) + (0.15 * s_unidad) + (0.30 * s_costo) + (0.15 * s_peso)
    score += bonus_marca(fila_factura["Producto_norm"], fila_maestro["Producto_norm"])
    return float(score)

def recuperar_candidatos(
    fila_factura: pd.Series,
    maestro: pd.DataFrame,
    top_n: int = 30,
    permitir_fallback_global: bool = True,
) -> pd.DataFrame:
    candidatos = maestro[
        maestro["RucProveedor"].astype(str) == str(fila_factura["RucProveedor"])
    ].copy()

    origen = "MISMO_PROVEEDOR"

    if candidatos.empty and permitir_fallback_global:
        candidatos = maestro.copy()
        origen = "GLOBAL"

    if candidatos.empty:
        return candidatos

    misma_unidad = candidatos[candidatos["Unidad_norm"] == fila_factura["Unidad_norm"]]
    if len(misma_unidad) >= min(10, len(candidatos)):
        candidatos = misma_unidad.copy()

    candidatos["heuristica"] = candidatos.apply(
        lambda r: score_heuristico(fila_factura, r),
        axis=1,
    )
    candidatos["OrigenCandidato"] = origen

    candidatos = (
        candidatos
        .sort_values("heuristica", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return candidatos
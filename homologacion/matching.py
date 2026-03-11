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
    claves = [
        "PANTENE", "HEAD", "SHOULDERS", "DOWNY", "ARIEL", "GILLETTE",
        "OLD", "SPICE", "SECRET", "ORAL", "B", "VICK", "PAMPERS",
        "VENUS", "ACE", "PRINCESS", "SPIDERMAN",
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

    diff_costo = abs(fila_factura["Costo_log"] - fila_maestro["Costo_log"])
    s_costo = float(np.exp(-2.0 * diff_costo))

    cod_fact = fila_factura["CodProducto"]
    score_cod = 0.0

    for c in ["CodProducto", "CodProducto2", "CodProducto3"]:
        cod_m = fila_maestro.get(c, "")
        if cod_fact and cod_m:
            if cod_fact[-6:] == cod_m[-6:]:
                score_cod = max(score_cod, 0.30)
            elif cod_fact[-5:] == cod_m[-5:]:
                score_cod = max(score_cod, 0.20)
            elif cod_fact[-4:] == cod_m[-4:]:
                score_cod = max(score_cod, 0.10)

    score = 0.55 * s_texto + 0.20 * s_unidad + 0.20 * s_costo + 0.05 * score_cod
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
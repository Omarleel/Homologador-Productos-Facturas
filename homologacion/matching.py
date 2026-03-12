import math
from difflib import SequenceMatcher
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


def token_set(texto: str) -> set[str]:
    return set(normalizar_texto(texto).split())


def jaccard(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / max(len(sa | sb), 1)


def similitud_log(a: float, b: float, escala: float = 1.0) -> float:
    return float(np.exp(-escala * abs(float(a) - float(b))))


def _rel_diff(a: float, b: float) -> float:
    a = float(a or 0.0)
    b = float(b or 0.0)

    if a <= 0.0 or b <= 0.0:
        return 1.0

    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def _sim_rel(a: float, b: float, escala: float = 6.0) -> float:
    if float(a or 0.0) <= 0.0 or float(b or 0.0) <= 0.0:
        return 0.5
    return float(math.exp(-escala * _rel_diff(a, b)))

FAMILY_STOPWORDS = {
    "UND", "UNIDAD", "UNIDADES", "CAJA", "CJA", "CJ", "PAQUETE", "PCK",
    "PACK", "PQT", "PAQ", "PCK", "BOL", "BOLSA", "BOT", "BANDEJA",
    "X", "BONIF", "CODIGO", "PRODUCTO", "PRODUCTOS",
    "SB", "AD", "PE", "EX", "REG",
}


def _tokens_familia(texto: str) -> list[str]:
    toks = [t for t in normalizar_texto(texto).split() if t]
    salida = []
    for t in toks:
        if t in FAMILY_STOPWORDS:
            continue
        if t.isdigit():
            continue
        if len(t) <= 2:
            continue
        salida.append(t)
    return salida


def _features_familia(fila_factura: pd.Series, fila_maestro: pd.Series) -> dict:
    fact = str(fila_factura.get("Producto_base_norm", "") or "")
    mast = str(fila_maestro.get("Producto_base_norm", "") or "")

    tf = _tokens_familia(fact)
    tm = _tokens_familia(mast)

    sf = set(tf)
    sm = set(tm)
    inter = sf & sm
    union = sf | sm

    overlap_count = len(inter)
    family_jaccard = (len(inter) / max(len(union), 1)) if sf and sm else 0.0

    same_anchor = 1.0 if tf and tm and tf[0] == tm[0] else 0.0
    same_prefix2 = 1.0 if len(tf) >= 2 and len(tm) >= 2 and tf[:2] == tm[:2] else 0.0

    return {
        "family_overlap_count": overlap_count,
        "family_jaccard": float(family_jaccard),
        "same_anchor": float(same_anchor),
        "same_prefix2": float(same_prefix2),
    }


def _same_family_strict(row: pd.Series) -> bool:
    return bool(
        row["same_prefix2"] == 1.0
        or row["family_overlap_count"] >= 2
        or (row["same_anchor"] == 1.0 and row["family_overlap_count"] >= 1)
    )


def _same_family_soft(row: pd.Series) -> bool:
    return bool(
        _same_family_strict(row)
        or (row["same_anchor"] == 1.0 and row["heuristica_texto"] >= 0.30)
        or (row["family_overlap_count"] >= 1 and row["heuristica_texto"] >= 0.34)
        or (row["heuristica_texto"] >= 0.72)
    )

def _primeros_tokens(texto: str, n: int = 3) -> Tuple[str, ...]:
    toks = [t for t in normalizar_texto(texto).split() if t]
    return tuple(toks[:n])


def _overlap_tokens(a: str, b: str) -> int:
    return len(token_set(a) & token_set(b))


def bonus_marca(fact_text: str, master_text: str) -> float:
    claves = [
        "BUBBALOO", "CHICLETS", "CLORETS", "CLUB SOCIAL", "FIELD", "HALLS", "RITZ", "TRIDENT",
        "ARUBA", "BELLA HOLANDESA", "BONLE", "CARTAVIO", "CASA GRANDE", "CHICOLAC", "COMPLETE",
        "DOLCCI", "DUMMY", "ECOLAT", "GLORIA", "LA FLORENCIA", "LA MESA", "LA PRADERA", "LEAF TEA",
        "MILKITO", "PRO", "PURA VIDA", "SAN MATEO", "SHAKE", "SOY VIDA", "TAMPICO", "YOFRESH",
        "YOMOST", "ZEROLACTO", "BARBIE", "CARS", "CLASICA TAN", "CLASICA URBANA", "HAPPY 1",
        "MAIS LIKE", "MAIS TRANCE", "PRINCESA", "SALOME", "SPIDERMAN", "URBANA", "BEIERSDORF",
        "COLGATE", "CREST", "DERMEX", "EFILA", "FAPE", "GENOMA LAB", "GOOD BRANDS", "GSK",
        "HERSIL", "INCASUR", "INTRADEVCO", "LEP", "MARUCHAN", "NINET", "NIVEA", "P&G", "PANASONIC",
        "RECKITT", "NOEL", "TMLUC", "PARACAS", "PETALO", "SEDITA", "ALACENA", "ALIANZA", "AMARAS",
        "AMOR", "ANGEL", "AVAL", "BLANCA FLOR", "BOLIVAR", "CAPRI", "CASINO", "COCINERO", "DENTO",
        "DIA", "DON VITTORIO", "ESPIGA DE ORO", "GEOMEN", "JUMBO", "LA FAVORITA", "LAVAGGI",
        "MARSELLA", "NICOLINI", "NORCHEFF", "OPAL", "PATITO", "PREMIO", "PRIMOR", "SAPOLIO",
        "SAYON", "SELLO DE ORO", "TROME", "UMSHA", "VICTORIA", "MIMASKOT", "3 OSITOS", "COSTA",
        "CAÑONAZO", "CAREZZA", "MOLITALIA", "MARCO POLO", "POMAROLA", "FANNY", "NUTRICAN",
        "PASCUALINO", "TODINNO", "VIZZIO", "CARICIA", "MONCLER", "ÑA PANCHA", "DON MANOLO",
        "FORTUNA", "MANPAN", "PALMEROLA", "POPEYE", "SPA", "TONDERO", "ACE", "ALWAYS", "ARIEL",
        "AYUDIN", "CLEARBLUE", "DOWNY", "GILLETTE", "HEPABIONTA", "HERBAL ESSENCES", "H&S",
        "OLD SPICE", "ORAL B", "PAMPERS", "PANTENE", "SECRET", "VICK", "BANDIDO", "CANBO",
        "FRESHCAN", "MICHICAT", "REX", "RICOCAN", "RICOCAT", "RICOCRACK", "SUPERCAN", "SUPERCAT",
        "THOR", "YAMIS DOG", "VUSE", "INKA KOLA", "CUSQUEÑA", "CRISTAL", "PILSEN CALLAO", "SUBLIME",
        "DONOFRIO", "TRIANGULO", "MOROCHAS", "CHARADA", "PIQUEO SNAX", "LAY'S", "DORITOS", "CHEETOS", 
        "CUATES", "CIELO", "PULP", "AJE", "VOLT", "SPORADE", "SABIDIA", "CUMANÁ", "WINTERS", "GN", 
        "BOLTS", "CUA CUA",     "KIRKLAND", "MAGGI", "NESTLÉ", "NESCAFÉ", "MILO", "IDEAL", "LA LECHERA", "KIRITOS",
        "COCA COLA", "FRUGOS", "SPRITE", "FANTA", "POWERADE", "AQUAFINA", "PEPSI", "SEVEN UP",
        "GATORADE", "CONCORDIA", "GUARANA", "SAN CARLOS", "CORONA", "BRAHMA", "PILSEN TRUJILLO",
        "AREQUIPEÑA", "PILSEN POLAR", "FREE TEA", "BIO AMARANTE", "KR", "ORO", "BIG COLA",
        "CHITOS", "CHEESE TRIS", "TORTEES", "MANI MOTO", "KARINTO", "FRITO LAY",
        "HUGGIES", "KOTEX", "SUAVE", "PLENIT", "SCOTT", "ELITE", "NOBLE", "BABYSEC",
        "LADYSOFT", "POISE", "TENA", "REDOXON", "APRILIS", "UMSHA", "BAMBINO",
        "ANITA", "COSTEÑO", "PAISANA", "VALLE NORTE", "HOJALMAR", "GRETEL", "CHINCHIN",
        "GRANJA AZUL", "LA CALERA", "HUACARIZ", "VIGOR", "DANLAC", "TAYTA",
        "TODINNO", "BAUDUCCO", "MOTTA", "BLANCA FLOR", "SAYON", "DORINA", "MANTEY",
        "LA DANESA", "MANTY", "STEVIA", "SUCRALOSA", "GLADE", "POETT", "CLOROX",
        "PINO", "ZORRITO", "PATITO", "SILJET", "RAID", "OFF", "CERINI", "VAPO"
    ]

    ft = token_set(fact_text)
    mt = token_set(master_text)

    puntos = 0.0
    for k in claves:
        if k in ft and k in mt:
            puntos += 0.04

    return min(puntos, 0.16)


def score_heuristico(fila_factura: pd.Series, fila_maestro: pd.Series) -> float:
    """
    Heurística textual/familiar.
    OJO: aquí NO dejamos que la presentación domine.
    """
    fact_base = str(fila_factura.get("Producto_base_norm", "") or "")
    mast_base = str(fila_maestro.get("Producto_base_norm", "") or "")

    fact_limpio = str(fila_factura.get("Producto_limpio", fact_base) or fact_base)
    mast_limpio = str(fila_maestro.get("Producto_limpio", mast_base) or mast_base)

    s_base_j = jaccard(fact_base, mast_base)
    s_clean_j = jaccard(fact_limpio, mast_limpio)
    s_seq = SequenceMatcher(None, normalizar_texto(fact_base), normalizar_texto(mast_base)).ratio()

    pfx_f = _primeros_tokens(fact_base, n=3)
    pfx_m = _primeros_tokens(mast_base, n=3)
    same_prefix = 1.0 if pfx_f and pfx_m and pfx_f[:2] == pfx_m[:2] else 0.0

    overlap = _overlap_tokens(fact_base, mast_base)
    overlap_score = min(overlap / 4.0, 1.0)

    s_unidad = 1.0 if fila_factura["Unidad_norm"] == fila_maestro["Unidad_norm"] else 0.0
    s_costo = similitud_log(fila_factura["Costo_log"], fila_maestro["Costo_log"], escala=2.2)

    score = (
        0.34 * s_base_j
        + 0.24 * s_seq
        + 0.14 * s_clean_j
        + 0.14 * overlap_score
        + 0.08 * same_prefix
        + 0.03 * s_unidad
        + 0.03 * s_costo
    )

    score += bonus_marca(fact_base, mast_base)
    return float(min(max(score, 0.0), 1.0))


def score_presentacion(fila_factura: pd.Series, fila_maestro: pd.Series) -> float:
    """
    Score de presentación normalizado en [0, 1].
    """
    tipo_f = str(fila_factura.get("TipoContenido", "NONE"))
    tipo_m = str(fila_maestro.get("TipoContenido", "NONE"))

    factor_f = float(fila_factura.get("FactorConversion", 0.0) or 0.0)
    factor_m = float(fila_maestro.get("FactorConversion", 0.0) or 0.0)

    cont_f = float(fila_factura.get("ContenidoUnidad", 0.0) or 0.0)
    cont_m = float(fila_maestro.get("ContenidoUnidad", 0.0) or 0.0)

    total_f = float(fila_factura.get("ContenidoTotal", 0.0) or 0.0)
    total_m = float(fila_maestro.get("ContenidoTotal", 0.0) or 0.0)

    peso_f = float(fila_factura.get("PesoUnitario", 0.0) or 0.0)
    peso_m = float(fila_maestro.get("PesoUnitario", 0.0) or 0.0)

    if tipo_f != "NONE" and tipo_m != "NONE":
        type_score = 1.0 if tipo_f == tipo_m else 0.0
    else:
        type_score = 0.65

    factor_score = _sim_rel(factor_f, factor_m, escala=7.0)
    content_score = _sim_rel(cont_f, cont_m, escala=8.0)
    total_score = _sim_rel(total_f, total_m, escala=8.0)
    peso_score = _sim_rel(peso_f, peso_m, escala=8.0)

    score = (
        0.20 * type_score
        + 0.22 * factor_score
        + 0.20 * content_score
        + 0.23 * total_score
        + 0.15 * peso_score
    )

    return float(min(max(score, 0.0), 1.0))


def tier_presentacion(fila_factura: pd.Series, fila_maestro: pd.Series) -> int:
    tipo_f = str(fila_factura.get("TipoContenido", "NONE"))
    tipo_m = str(fila_maestro.get("TipoContenido", "NONE"))

    factor_rd = _rel_diff(
        fila_factura.get("FactorConversion", 0.0),
        fila_maestro.get("FactorConversion", 0.0),
    )
    content_rd = _rel_diff(
        fila_factura.get("ContenidoUnidad", 0.0),
        fila_maestro.get("ContenidoUnidad", 0.0),
    )
    total_rd = _rel_diff(
        fila_factura.get("ContenidoTotal", 0.0),
        fila_maestro.get("ContenidoTotal", 0.0),
    )
    peso_rd = _rel_diff(
        fila_factura.get("PesoUnitario", 0.0),
        fila_maestro.get("PesoUnitario", 0.0),
    )

    mismo_tipo = (tipo_f == tipo_m) or (tipo_f == "NONE") or (tipo_m == "NONE")
    mismo_factor = factor_rd <= 0.02
    contenido_muy_cerca = content_rd <= 0.08
    total_muy_cerca = total_rd <= 0.08
    peso_muy_cerca = peso_rd <= 0.08

    if mismo_tipo and mismo_factor and (contenido_muy_cerca or total_muy_cerca or peso_muy_cerca):
        return 0

    if mismo_tipo and (content_rd <= 0.15 or total_rd <= 0.15 or peso_rd <= 0.15):
        return 1

    if mismo_tipo and factor_rd <= 0.10:
        return 2

    if mismo_tipo:
        return 3

    return 4


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

    candidatos["heuristica_texto"] = candidatos.apply(
        lambda r: score_heuristico(fila_factura, r),
        axis=1,
    )

    shortlist_n = max(top_n * 12, 120)
    candidatos = (
        candidatos
        .sort_values("heuristica_texto", ascending=False)
        .head(shortlist_n)
        .copy()
    )

    fam = candidatos.apply(
        lambda r: pd.Series(_features_familia(fila_factura, r)),
        axis=1,
    )
    candidatos = pd.concat([candidatos.reset_index(drop=True), fam.reset_index(drop=True)], axis=1)

    candidatos["same_family_strict"] = candidatos.apply(_same_family_strict, axis=1)
    candidatos["same_family_soft"] = candidatos.apply(_same_family_soft, axis=1)

    fam_strict = candidatos[candidatos["same_family_strict"]].copy()
    if len(fam_strict) >= max(top_n, 5):
        candidatos = fam_strict
    else:
        fam_soft = candidatos[candidatos["same_family_soft"]].copy()
        if len(fam_soft) >= max(top_n, 5):
            candidatos = fam_soft

    candidatos["score_presentacion"] = candidatos.apply(
        lambda r: score_presentacion(fila_factura, r),
        axis=1,
    )

    candidatos["tier_presentacion"] = candidatos.apply(
        lambda r: tier_presentacion(fila_factura, r),
        axis=1,
    )

    candidatos["factor_match_strict"] = candidatos.apply(
        lambda r: _factor_match_strict(fila_factura, r),
        axis=1,
    )

    candidatos["content_match_strict"] = candidatos.apply(
        lambda r: _content_match_strict(fila_factura, r),
        axis=1,
    )

    candidatos["total_match_strict"] = candidatos.apply(
        lambda r: _total_match_strict(fila_factura, r),
        axis=1,
    )

    candidatos["peso_match_strict"] = candidatos.apply(
        lambda r: _peso_match_strict(fila_factura, r),
        axis=1,
    )

    candidatos["tipo_match_strict"] = candidatos.apply(
        lambda r: _tipo_match_strict(fila_factura, r),
        axis=1,
    )

    candidatos["presentacion_strict"] = (
        candidatos["same_family_soft"]
        & candidatos["tipo_match_strict"]
        & candidatos["factor_match_strict"]
        & (
            candidatos["content_match_strict"]
            | candidatos["total_match_strict"]
            | candidatos["peso_match_strict"]
        )
    )

    candidatos["presentacion_soft"] = (
        candidatos["same_family_soft"]
        & candidatos["tipo_match_strict"]
        & (
            candidatos["factor_match_strict"]
            | (candidatos["tier_presentacion"] <= 1)
        )
    )

    candidatos["lexical_gate"] = np.clip(
        (candidatos["heuristica_texto"] - 0.12) / 0.35,
        0.0,
        1.0,
    )

    candidatos["score_pre_modelo"] = (
        0.78 * candidatos["heuristica_texto"]
        + 0.22 * candidatos["score_presentacion"]
    ) * (0.20 + 0.80 * candidatos["lexical_gate"])

    candidatos["OrigenCandidato"] = origen

    estrictos = candidatos[candidatos["presentacion_strict"]].copy()
    if len(estrictos) >= top_n:
        candidatos = estrictos
    else:
        suaves = candidatos[candidatos["presentacion_soft"]].copy()
        if len(suaves) >= top_n:
            candidatos = suaves

    candidatos = (
        candidatos
        .sort_values(
            [
                "same_family_strict",
                "same_family_soft",
                "tier_presentacion",
                "presentacion_strict",
                "score_pre_modelo",
                "heuristica_texto",
            ],
            ascending=[False, False, True, False, False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return candidatos

def _factor_match_strict(fila_factura: pd.Series, fila_maestro: pd.Series) -> bool:
    ff = float(fila_factura.get("FactorConversion", 0.0) or 0.0)
    fm = float(fila_maestro.get("FactorConversion", 0.0) or 0.0)

    if ff <= 0.0 or fm <= 0.0:
        return False

    return _rel_diff(ff, fm) <= 0.02


def _content_match_strict(fila_factura: pd.Series, fila_maestro: pd.Series) -> bool:
    cf = float(fila_factura.get("ContenidoUnidad", 0.0) or 0.0)
    cm = float(fila_maestro.get("ContenidoUnidad", 0.0) or 0.0)

    if cf <= 0.0 or cm <= 0.0:
        return False

    return _rel_diff(cf, cm) <= 0.08


def _total_match_strict(fila_factura: pd.Series, fila_maestro: pd.Series) -> bool:
    tf = float(fila_factura.get("ContenidoTotal", 0.0) or 0.0)
    tm = float(fila_maestro.get("ContenidoTotal", 0.0) or 0.0)

    if tf <= 0.0 or tm <= 0.0:
        return False

    return _rel_diff(tf, tm) <= 0.08


def _peso_match_strict(fila_factura: pd.Series, fila_maestro: pd.Series) -> bool:
    pf = float(fila_factura.get("PesoUnitario", 0.0) or 0.0)
    pm = float(fila_maestro.get("PesoUnitario", 0.0) or 0.0)

    if pf <= 0.0 or pm <= 0.0:
        return False

    return _rel_diff(pf, pm) <= 0.08


def _tipo_match_strict(fila_factura: pd.Series, fila_maestro: pd.Series) -> bool:
    tf = str(fila_factura.get("TipoContenido", "NONE"))
    tm = str(fila_maestro.get("TipoContenido", "NONE"))

    if tf == "NONE" or tm == "NONE":
        return True

    return tf == tm
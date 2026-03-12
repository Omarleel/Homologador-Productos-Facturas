from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED
from .matching import construir_indice_codigos, jaccard, similitud_log
from .preparacion import preparar_facturas, preparar_maestro


def _build_maestro_por_ruc(maestro: pd.DataFrame) -> dict[str, pd.DataFrame]:
    maestro_tmp = maestro.copy()
    maestro_tmp["_ruc_key"] = maestro_tmp["RucProveedor"].astype(str).str.strip()

    maestro_por_ruc: dict[str, pd.DataFrame] = {}
    for ruc, group in maestro_tmp.groupby("_ruc_key", sort=False):
        maestro_por_ruc[ruc] = group.drop(columns=["_ruc_key"]).reset_index(drop=True)

    return maestro_por_ruc


def _cached_jaccard_factory():
    cache: dict[tuple[str, str], float] = {}

    def cached(a: str, b: str) -> float:
        key = (str(a), str(b))
        value = cache.get(key)
        if value is None:
            value = float(jaccard(a, b))
            cache[key] = value
        return value

    return cached


def _cached_similitud_log_factory():
    cache: dict[tuple[float, float, float], float] = {}

    def cached(a: float, b: float, escala: float) -> float:
        key = (float(a), float(b), float(escala))
        value = cache.get(key)
        if value is None:
            value = float(similitud_log(a, b, escala=escala))
            cache[key] = value
        return value

    return cached


def resolver_positivos_por_codigo(historial: pd.DataFrame, maestro: pd.DataFrame) -> pd.DataFrame:
    idx = construir_indice_codigos(maestro)
    filas = []

    for f in historial.itertuples(index=False):
        key = (str(f.RucProveedor).strip(), f.CodProducto)
        m_idx = idx.get(key)

        if m_idx is None:
            continue

        m = maestro.loc[m_idx]

        filas.append({
            "fact_cod": f.CodProducto,
            "fact_text": f.Producto_norm,
            "fact_base_text": f.Producto_base_norm,
            "fact_unit": f.Unidad_norm,
            "fact_type": f.TipoContenido,
            "fact_cost": f.Costo_log,
            "fact_peso": f.PesoUnitario,
            "fact_factor": f.Factor_log,
            "fact_content": f.ContenidoUnidad_log,
            "fact_total": f.ContenidoTotal_log,
            "master_cod": m["CodProducto"],
            "master_text": m["Producto_norm"],
            "master_base_text": m["Producto_base_norm"],
            "master_unit": m["Unidad_norm"],
            "master_type": m["TipoContenido"],
            "master_cost": m["Costo_log"],
            "master_peso": m["PesoUnitario"],
            "master_factor": m["Factor_log"],
            "master_content": m["ContenidoUnidad_log"],
            "master_total": m["ContenidoTotal_log"],
            "label": 1,
            "RucProveedor": f.RucProveedor,
        })

    return pd.DataFrame(filas)


def muestrear_negativos(
    positivos: pd.DataFrame,
    maestro: pd.DataFrame,
    n_neg_por_pos: int = 5,
) -> pd.DataFrame:
    negativos = []
    rng = np.random.default_rng(SEED)

    maestro_por_ruc = _build_maestro_por_ruc(maestro)
    cached_jaccard = _cached_jaccard_factory()
    cached_similitud_log = _cached_similitud_log_factory()

    for p in positivos.itertuples(index=False):
        ruc_key = str(p.RucProveedor).strip()
        maestro_ruc = maestro_por_ruc.get(ruc_key)

        if maestro_ruc is None or maestro_ruc.empty:
            continue

        pool = maestro_ruc[maestro_ruc["CodProducto"] != p.master_cod].copy()

        if pool.empty:
            continue

        fact_text = str(p.fact_text)
        fact_base_text = str(p.fact_base_text)
        fact_cost = float(p.fact_cost)
        fact_factor = float(p.fact_factor)
        fact_total = float(p.fact_total)
        fact_type = p.fact_type

        pool["sim_text"] = pool["Producto_norm"].map(
            lambda x: cached_jaccard(fact_text, str(x))
        )
        pool["sim_base"] = pool["Producto_base_norm"].map(
            lambda x: cached_jaccard(fact_base_text, str(x))
        )
        pool["sim_cost"] = pool["Costo_log"].map(
            lambda x: cached_similitud_log(fact_cost, float(x), 1.8)
        )
        pool["sim_factor"] = pool["Factor_log"].map(
            lambda x: cached_similitud_log(fact_factor, float(x), 2.4)
        )
        pool["sim_total"] = pool["ContenidoTotal_log"].map(
            lambda x: cached_similitud_log(fact_total, float(x), 1.8)
        )
        pool["same_type"] = (pool["TipoContenido"] == fact_type).astype(float)

        pool["hardness"] = (
            0.35 * pool["sim_text"]
            + 0.20 * pool["sim_base"]
            + 0.15 * pool["sim_cost"]
            + 0.10 * pool["sim_factor"]
            + 0.20 * (pool["sim_total"] * pool["same_type"])
        )

        top_hard = pool.nlargest(max(n_neg_por_pos * 4, 10), columns="hardness")

        n_hard = min(len(top_hard), max(1, int(round(n_neg_por_pos * 0.7))))
        n_easy = max(0, n_neg_por_pos - n_hard)

        if len(top_hard) > n_hard:
            hard_sel = top_hard.iloc[
                rng.choice(len(top_hard), size=n_hard, replace=False)
            ]
        else:
            hard_sel = top_hard

        used = set(hard_sel.index)
        remaining = pool.loc[~pool.index.isin(used)]

        if n_easy > 0 and not remaining.empty:
            easy_sel = remaining.sample(
                min(n_easy, len(remaining)),
                random_state=SEED,
            )
            seleccionados = pd.concat([hard_sel, easy_sel], ignore_index=True)
        else:
            seleccionados = hard_sel.reset_index(drop=True)

        seleccionados = seleccionados.drop_duplicates(subset=["CodProducto"]).head(n_neg_por_pos)

        for m in seleccionados.itertuples(index=False):
            negativos.append({
                "fact_cod": p.fact_cod,
                "fact_text": p.fact_text,
                "fact_base_text": p.fact_base_text,
                "fact_unit": p.fact_unit,
                "fact_type": p.fact_type,
                "fact_cost": p.fact_cost,
                "fact_peso": p.fact_peso,
                "fact_factor": p.fact_factor,
                "fact_content": p.fact_content,
                "fact_total": p.fact_total,
                "master_cod": m.CodProducto,
                "master_text": m.Producto_norm,
                "master_base_text": m.Producto_base_norm,
                "master_unit": m.Unidad_norm,
                "master_type": m.TipoContenido,
                "master_cost": m.Costo_log,
                "master_peso": m.PesoUnitario,
                "master_factor": m.Factor_log,
                "master_content": m.ContenidoUnidad_log,
                "master_total": m.ContenidoTotal_log,
                "label": 0,
                "RucProveedor": p.RucProveedor,
            })

    return pd.DataFrame(negativos)


def construir_dataset_entrenamiento(
    maestro: pd.DataFrame,
    historial_facturas: pd.DataFrame,
    n_neg_por_pos: int = 5,
) -> pd.DataFrame:
    maestro_p = preparar_maestro(maestro)
    hist_p = preparar_facturas(historial_facturas)

    positivos = resolver_positivos_por_codigo(hist_p, maestro_p)

    if positivos.empty:
        raise ValueError(
            "No se generaron positivos automáticos. Necesitas facturas históricas "
            "con coincidencias exactas entre el código de factura y alguno de los "
            "3 códigos del maestro."
        )

    negativos = muestrear_negativos(positivos, maestro_p, n_neg_por_pos=n_neg_por_pos)

    pares = pd.concat([positivos, negativos], ignore_index=True)
    pares = pares.drop_duplicates(
        subset=[
            "fact_cod",
            "master_cod",
            "fact_text",
            "master_text",
            "label",
        ]
    )
    pares = pares.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return pares
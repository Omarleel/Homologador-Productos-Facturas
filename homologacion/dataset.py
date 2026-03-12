import numpy as np
import pandas as pd

from .config import SEED
from .matching import construir_indice_codigos, jaccard, similitud_log
from .preparacion import preparar_facturas, preparar_maestro


def resolver_positivos_por_codigo(historial: pd.DataFrame, maestro: pd.DataFrame) -> pd.DataFrame:
    idx = construir_indice_codigos(maestro)
    filas = []

    for _, f in historial.iterrows():
        key = (str(f["RucProveedor"]).strip(), f["CodProducto"])
        m_idx = idx.get(key)

        if m_idx is None:
            continue

        m = maestro.loc[m_idx]

        filas.append({
            "fact_cod": f["CodProducto"],
            "fact_text": f["Producto_norm"],
            "fact_base_text": f["Producto_base_norm"],
            "fact_unit": f["Unidad_norm"],
            "fact_type": f["TipoContenido"],
            "fact_cost": f["Costo_log"],
            "fact_peso": f["PesoUnitario"],
            "fact_factor": f["Factor_log"],
            "fact_content": f["ContenidoUnidad_log"],
            "fact_total": f["ContenidoTotal_log"],
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
            "RucProveedor": f["RucProveedor"],
        })

    return pd.DataFrame(filas)


def muestrear_negativos(
    positivos: pd.DataFrame,
    maestro: pd.DataFrame,
    n_neg_por_pos: int = 5,
) -> pd.DataFrame:
    negativos = []
    rng = np.random.default_rng(SEED)

    for _, p in positivos.iterrows():
        pool = maestro[
            (maestro["RucProveedor"].astype(str) == str(p["RucProveedor"])) &
            (maestro["CodProducto"] != p["master_cod"])
        ].copy()

        if pool.empty:
            continue

        pool["sim_text"] = pool["Producto_norm"].apply(lambda x: jaccard(p["fact_text"], x))
        pool["sim_base"] = pool["Producto_base_norm"].apply(lambda x: jaccard(p["fact_base_text"], x))
        pool["sim_cost"] = pool["Costo_log"].apply(lambda x: similitud_log(p["fact_cost"], x, escala=1.8))
        pool["sim_factor"] = pool["Factor_log"].apply(lambda x: similitud_log(p["fact_factor"], x, escala=2.4))
        pool["sim_total"] = pool["ContenidoTotal_log"].apply(lambda x: similitud_log(p["fact_total"], x, escala=1.8))
        pool["same_type"] = (pool["TipoContenido"] == p["fact_type"]).astype(float)

        pool["hardness"] = (
            0.35 * pool["sim_text"]
            + 0.20 * pool["sim_base"]
            + 0.15 * pool["sim_cost"]
            + 0.10 * pool["sim_factor"]
            + 0.20 * (pool["sim_total"] * pool["same_type"])
        )

        top_hard = pool.sort_values("hardness", ascending=False).head(max(n_neg_por_pos * 4, 10))
        n_hard = min(len(top_hard), max(1, int(round(n_neg_por_pos * 0.7))))
        n_easy = max(0, n_neg_por_pos - n_hard)

        if len(top_hard) > n_hard:
            hard_sel = top_hard.iloc[rng.choice(len(top_hard), size=n_hard, replace=False)]
        else:
            hard_sel = top_hard

        used = set(hard_sel.index)
        remaining = pool.loc[~pool.index.isin(used)]

        if n_easy > 0 and not remaining.empty:
            easy_sel = remaining.sample(min(n_easy, len(remaining)), random_state=SEED)
            seleccionados = pd.concat([hard_sel, easy_sel], ignore_index=True)
        else:
            seleccionados = hard_sel.reset_index(drop=True)

        # Evitar duplicados por código.
        seleccionados = seleccionados.drop_duplicates(subset=["CodProducto"]).head(n_neg_por_pos)

        for _, m in seleccionados.iterrows():
            negativos.append({
                "fact_cod": p["fact_cod"],
                "fact_text": p["fact_text"],
                "fact_base_text": p["fact_base_text"],
                "fact_unit": p["fact_unit"],
                "fact_type": p["fact_type"],
                "fact_cost": p["fact_cost"],
                "fact_peso": p["fact_peso"],
                "fact_factor": p["fact_factor"],
                "fact_content": p["fact_content"],
                "fact_total": p["fact_total"],
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
                "label": 0,
                "RucProveedor": p["RucProveedor"],
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
            "fact_cod", "master_cod", "fact_text", "master_text", "label",
        ]
    )
    pares = pares.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return pares
import numpy as np
import pandas as pd

from .config import SEED
from .matching import construir_indice_codigos
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
            "fact_unit": f["Unidad_norm"],
            "fact_cost": f["Costo_log"],
            "master_cod": m["CodProducto"],
            "master_text": m["Producto_norm"],
            "master_unit": m["Unidad_norm"],
            "master_cost": m["Costo_log"],
            "label": 1,
            "RucProveedor": f["RucProveedor"],
        })

    return pd.DataFrame(filas)


def muestrear_negativos(
    positivos: pd.DataFrame,
    maestro: pd.DataFrame,
    n_neg_por_pos: int = 4,
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

        pool["delta_costo"] = (pool["Costo_log"] - p["fact_cost"]).abs()

        misma_unidad = pool[pool["Unidad_norm"] == p["fact_unit"]].sort_values("delta_costo")
        candidatos = misma_unidad.head(n_neg_por_pos * 3)

        if len(candidatos) < n_neg_por_pos:
            faltan = n_neg_por_pos - len(candidatos)
            extra = pool.sample(min(faltan * 3, len(pool)), random_state=SEED)
            candidatos = (
                pd.concat([candidatos, extra], ignore_index=True)
                .drop_duplicates(subset=["CodProducto"])
            )

        if len(candidatos) > n_neg_por_pos:
            idxs = rng.choice(len(candidatos), size=n_neg_por_pos, replace=False)
            candidatos = candidatos.iloc[idxs]

        for _, m in candidatos.iterrows():
            negativos.append({
                "fact_cod": p["fact_cod"],
                "fact_text": p["fact_text"],
                "fact_unit": p["fact_unit"],
                "fact_cost": p["fact_cost"],
                "master_cod": m["CodProducto"],
                "master_text": m["Producto_norm"],
                "master_unit": m["Unidad_norm"],
                "master_cost": m["Costo_log"],
                "label": 0,
                "RucProveedor": p["RucProveedor"],
            })

    return pd.DataFrame(negativos)


def construir_dataset_entrenamiento(
    maestro: pd.DataFrame,
    historial_facturas: pd.DataFrame,
    n_neg_por_pos: int = 4,
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
    pares = pares.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    return pares
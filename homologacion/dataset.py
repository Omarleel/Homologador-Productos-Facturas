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
            "fact_peso": f["PesoUnitario"],                # peso real
            "master_cod": m["CodProducto"],
            "master_text": m["Producto_norm"],
            "master_unit": m["Unidad_norm"],
            "master_cost": m["Costo_log"],
            "master_peso": m["PesoUnitario"],              # peso real
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

    # Distribución de los tipos de negativos
    if n_neg_por_pos >= 3:
        n_hard_text = 1
        n_hard_cost = 1
        n_hard_peso = 1
        n_easy = n_neg_por_pos - 3
    else:
        # Para valores pequeños, se reparten equitativamente (redondeo)
        n_hard_text = max(1, n_neg_por_pos // 3)
        n_hard_cost = max(1, n_neg_por_pos // 3)
        n_hard_peso = max(1, n_neg_por_pos // 3)
        n_easy = n_neg_por_pos - n_hard_text - n_hard_cost - n_hard_peso
        if n_easy < 0:
            # Ajuste si la suma excede
            n_hard_text = n_neg_por_pos // 3
            n_hard_cost = n_neg_por_pos // 3
            n_hard_peso = n_neg_por_pos - n_hard_text - n_hard_cost
            n_easy = 0

    for _, p in positivos.iterrows():
        pool = maestro[
            (maestro["RucProveedor"].astype(str) == str(p["RucProveedor"])) &
            (maestro["CodProducto"] != p["master_cod"])
        ].copy()

        if pool.empty:
            continue

        # Calcular similitud textual (Jaccard)
        fact_tokens = set(p["fact_text"].split())
        pool["jaccard"] = pool["Producto_norm"].apply(
            lambda x: len(fact_tokens & set(x.split())) / max(len(fact_tokens | set(x.split())), 1)
        )

        # Diferencias de costo y peso
        pool["delta_costo"] = (pool["Costo_log"] - p["fact_cost"]).abs()
        pool["delta_peso"] = (pool["PesoUnitario"] - p["fact_peso"]).abs()

        # Normalizadores para combinar puntajes
        max_j = pool["jaccard"].max() if pool["jaccard"].max() > 0 else 1.0
        max_c = pool["delta_costo"].max() if pool["delta_costo"].max() > 0 else 1.0
        max_p = pool["delta_peso"].max() if pool["delta_peso"].max() > 0 else 1.0

        # Puntajes compuestos para cada tipo de dureza
        # - Texto duro: alta similitud textual
        pool["score_text_hard"] = pool["jaccard"] / max_j

        # - Costo duro: baja similitud textual + alta diferencia de costo
        pool["score_cost_hard"] = (1 - pool["jaccard"] / max_j) + (pool["delta_costo"] / max_c)

        # - Peso duro: alta similitud textual + alta diferencia de peso
        pool["score_peso_hard"] = (pool["jaccard"] / max_j) + (pool["delta_peso"] / max_p)

        # Separar por unidad (los duros se toman preferentemente de la misma unidad)
        misma_unidad = pool[pool["Unidad_norm"] == p["fact_unit"]]
        distinta_unidad = pool[pool["Unidad_norm"] != p["fact_unit"]]

        # 1. Negativos duros por texto
        if not misma_unidad.empty:
            text_cand = misma_unidad.nlargest(n_hard_text * 3, "score_text_hard")
            if len(text_cand) > n_hard_text:
                text_hard = text_cand.iloc[rng.choice(len(text_cand), size=n_hard_text, replace=False)]
            else:
                text_hard = text_cand
        else:
            text_hard = pd.DataFrame()

        # 2. Negativos duros por costo
        if not misma_unidad.empty:
            cost_cand = misma_unidad.nlargest(n_hard_cost * 3, "score_cost_hard")
            if len(cost_cand) > n_hard_cost:
                cost_hard = cost_cand.iloc[rng.choice(len(cost_cand), size=n_hard_cost, replace=False)]
            else:
                cost_hard = cost_cand
        else:
            cost_hard = pd.DataFrame()

        # 3. Negativos duros por peso
        if not misma_unidad.empty:
            peso_cand = misma_unidad.nlargest(n_hard_peso * 3, "score_peso_hard")
            if len(peso_cand) > n_hard_peso:
                peso_hard = peso_cand.iloc[rng.choice(len(peso_cand), size=n_hard_peso, replace=False)]
            else:
                peso_hard = peso_cand
        else:
            peso_hard = pd.DataFrame()

        # 4. Negativos fáciles (resto, incluyendo distinta unidad)
        used_indices = set(text_hard.index) | set(cost_hard.index) | set(peso_hard.index)
        remaining = pool.loc[~pool.index.isin(used_indices)]
        if len(remaining) > n_easy:
            easy = remaining.sample(n_easy, random_state=SEED)
        else:
            easy = remaining

        # Combinar todos los seleccionados
        seleccionados = pd.concat([text_hard, cost_hard, peso_hard, easy], ignore_index=True)

        # Si aún faltan, completar con muestreo aleatorio de todo el pool
        if len(seleccionados) < n_neg_por_pos:
            faltan = n_neg_por_pos - len(seleccionados)
            extra = pool.sample(min(faltan, len(pool)), replace=False, random_state=SEED)
            seleccionados = pd.concat([seleccionados, extra], ignore_index=True)

        # Agregar a la lista de negativos
        for _, m in seleccionados.iterrows():
            negativos.append({
                "fact_cod": p["fact_cod"],
                "fact_text": p["fact_text"],
                "fact_unit": p["fact_unit"],
                "fact_cost": p["fact_cost"],
                "fact_peso": p["fact_peso"],
                "master_cod": m["CodProducto"],
                "master_text": m["Producto_norm"],
                "master_unit": m["Unidad_norm"],
                "master_cost": m["Costo_log"],
                "master_peso": m["PesoUnitario"],
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
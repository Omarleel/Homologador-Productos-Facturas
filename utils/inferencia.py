from typing import Optional
import numpy as np
import pandas as pd

from .matching import (
    buscar_match_exacto,
    construir_indice_codigos,
    recuperar_candidatos,
    calc_same_brand,
    calc_brand_conflict
    
)
from core.model import ModeloMatchProducto
from .preparacion import preparar_facturas, preparar_maestro

def inferir_codproducto(
    facturas_nuevas: pd.DataFrame,
    maestro: pd.DataFrame,
    modelo_match: ModeloMatchProducto,
    top_k: int = 3,
    umbral_match: Optional[float] = None,
    top_n_candidatos: int = 40,
) -> pd.DataFrame:
    if umbral_match is None:
        umbral_match = getattr(modelo_match, "best_threshold", 0.72)

    maestro_p = preparar_maestro(maestro)
    fact_p = preparar_facturas(facturas_nuevas)
    idx = construir_indice_codigos(maestro_p)

    resultados = []

    for _, f in fact_p.iterrows():
        
        exacto = buscar_match_exacto(f, maestro_p, idx)

        if exacto is not None:
            row = exacto.to_dict()
            row.update({
                "OrigenCandidato": "EXACTO",
                "TipoResultado": "EXACTO",
                "Score": 1.0,
                "CodFactura": f["CodProducto"],
                "ProductoFactura": f["Producto"],
                "ProductoFacturaBase": f["Producto_base_norm"],
                "UnidadFactura": f["UnidaMedidaCompra"],
                "CostoFactura": f["CostoCaja"],
                "FactorFactura": f["FactorConversion"],
                "ContenidoFactura": f["ContenidoUnidad"],
                "ContenidoTotalFactura": f["ContenidoTotal"],
                "TipoContenidoFactura": f["TipoContenido"],
                "Producto_norm_factura": f["Producto_norm"],
                "Unidad_norm_factura": f["Unidad_norm"],
                "Costo_log_factura": f["Costo_log"],
                "Rank": 1,
            })
            resultados.append(row)
            continue

        cand = recuperar_candidatos(f, maestro_p, top_n=top_n_candidatos)

        if cand.empty:
            resultados.append({
                "RucProveedor": f["RucProveedor"],
                "CodFactura": f["CodProducto"],
                "ProductoFactura": f["Producto"],
                "ProductoFacturaBase": f["Producto_base_norm"],
                "UnidadFactura": f["UnidaMedidaCompra"],
                "CostoFactura": f["CostoCaja"],
                "FactorFactura": f["FactorConversion"],
                "ContenidoFactura": f["ContenidoUnidad"],
                "ContenidoTotalFactura": f["ContenidoTotal"],
                "TipoContenidoFactura": f["TipoContenido"],
                "TipoResultado": "SIN_CANDIDATOS",
                "Score": 0.0,
                "Rank": 1,
            })
            continue

        pares = pd.DataFrame({
            "fact_text": [f["Producto_norm"]] * len(cand),
            "fact_base_text": [f["Producto_base_norm"]] * len(cand),
            "fact_unit": [f["Unidad_norm"]] * len(cand),
            "fact_type": [f["TipoContenido"]] * len(cand),
            "fact_cost": [f["Costo_log"]] * len(cand),
            "fact_peso": [f["PesoUnitario"]] * len(cand),
            "fact_factor": [f["Factor_log"]] * len(cand),
            "fact_content": [f["ContenidoUnidad_log"]] * len(cand),
            "fact_total": [f["ContenidoTotal_log"]] * len(cand),
            "master_text": cand["Producto_norm"].values,
            "master_base_text": cand["Producto_base_norm"].values,
            "master_unit": cand["Unidad_norm"].values,
            "master_type": cand["TipoContenido"].values,
            "master_cost": cand["Costo_log"].values,
            "master_peso": cand["PesoUnitario"].values,
            "master_factor": cand["Factor_log"].values,
            "master_content": cand["ContenidoUnidad_log"].values,
            "master_total": cand["ContenidoTotal_log"].values,
            "label": [0] * len(cand),
        })

        probs = modelo_match.predict_pairs(pares)

        cand = cand.copy()
        cand["ScoreModelo"] = probs

        col_heur = "heuristica_texto" if "heuristica_texto" in cand.columns else "heuristica"

        cand["LexicalGate"] = np.clip(
            (cand[col_heur] - 0.12) / 0.35,
            0.0,
            1.0,
        )

        cand["family_bonus"] = (
            0.03 * cand["same_family_strict"].astype(float) +
            0.05 * cand["same_family_soft"].astype(float)
        )

        cand["same_brand"] = cand["Producto_base_norm"].apply(
            lambda t: calc_same_brand(f["Producto_base_norm"], t)
        )

        cand["brand_conflict"] = cand["Producto_base_norm"].apply(
            lambda t: calc_brand_conflict(f["Producto_base_norm"], t)
        )

        cand["estructura_ok"] = (
            cand["presentacion_soft"].astype(int)
            + cand["factor_match_strict"].astype(int)
            + cand["content_match_strict"].astype(int)
            + cand["total_match_strict"].astype(int)
        ) >= 2

        cand["brand_bonus"] = np.where(
            cand["estructura_ok"],
            0.06 * cand["same_brand"] - 0.10 * cand["brand_conflict"],
            0.02 * cand["same_brand"] - 0.12 * cand["brand_conflict"]
        )

        cand["presentation_penalty"] = (
            0.08 * (~cand["factor_match_strict"]).astype(float) +
            0.10 * (~cand["content_match_strict"]).astype(float) +
            0.10 * (~cand["total_match_strict"]).astype(float) +
            0.06 * (~cand["tipo_match_strict"]).astype(float)
        )

        cand["tipo_none_penalty"] = (
            (cand["TipoContenido"].astype(str) == "NONE").astype(float) * 0.05
        )

        cand["ScoreFinal"] = (
            0.56 * cand["ScoreModelo"]
            + 0.14 * cand[col_heur]
            + 0.18 * cand["score_presentacion"]
            + cand["family_bonus"]
            + cand["brand_bonus"]
            - cand["presentation_penalty"]
            - cand["tipo_none_penalty"]
        ) * (0.35 + 0.65 * cand["LexicalGate"])

        cand = cand.sort_values(
            ["ScoreFinal", "ScoreModelo", col_heur, "tier_presentacion"],
            ascending=[False, False, False, True],
        ).head(top_k)

        cand["Score"] = cand["ScoreFinal"]

        mejor = cand.iloc[0]
        tipo = "TENTATIVO" if mejor["Score"] >= umbral_match else "POSIBLE_NUEVO_PRODUCTO"

        for rank, (_, c) in enumerate(cand.iterrows(), start=1):
            row = c.drop(
                labels=["heuristica"],
                errors="ignore",
            ).to_dict()

            row.update({
                "TipoResultado": tipo if rank == 1 else "ALTERNATIVA",
                "CodFactura": f["CodProducto"],
                "ProductoFactura": f["Producto"],
                "ProductoFacturaBase": f["Producto_base_norm"],
                "UnidadFactura": f["UnidaMedidaCompra"],
                "CostoFactura": f["CostoCaja"],
                "FactorFactura": f["FactorConversion"],
                "ContenidoFactura": f["ContenidoUnidad"],
                "ContenidoTotalFactura": f["ContenidoTotal"],
                "TipoContenidoFactura": f["TipoContenido"],
                "Producto_norm_factura": f["Producto_norm"],
                "Unidad_norm_factura": f["Unidad_norm"],
                "Costo_log_factura": f["Costo_log"],
                "Rank": rank,
            })

            resultados.append(row)

    return pd.DataFrame(resultados)
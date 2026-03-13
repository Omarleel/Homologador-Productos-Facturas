from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils import construir_dataset_entrenamiento, init_seeds
from utils.config import (
    dataset_path,
    ensure_project_dirs,
    model_path,
    require_file,
)
from core.model import ModeloMatchProducto


MODELO_NOMBRE = "homologacion_v2"

ARCHIVO_MAESTRO = "maestro.csv"
ARCHIVO_HISTORIAL_INCREMENTAL = "historial_facturas.csv" # Para reentrenar necesitamos un histórico ya homologado
ARCHIVO_PARES_REENTRENAMIENTO = "pares_reentrenamiento_incremental.csv"
ARCHIVO_PARES_HISTORICOS = "pares_entrenamiento.csv"


def crear_backup_modelo(modelo_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_backup = modelo_dir.parent / f"{modelo_dir.name}_backup_{timestamp}"
    shutil.copytree(modelo_dir, ruta_backup)
    return ruta_backup


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def should_promote(
    champion: dict,
    challenger: dict,
    min_delta_pr_auc: float = 0.003,
    min_delta_f1: float = 0.005,
    max_precision_drop: float = 0.01,
    max_recall_drop: float = 0.02,
) -> tuple[bool, str]:
    d_pr = challenger["pr_auc"] - champion["pr_auc"]
    d_f1 = challenger["best_f1_eval"] - champion["best_f1_eval"]
    d_prec = challenger["best_precision_eval"] - champion["best_precision_eval"]
    d_rec = challenger["best_recall_eval"] - champion["best_recall_eval"]

    if d_pr >= min_delta_pr_auc and d_prec >= -max_precision_drop and d_rec >= -max_recall_drop:
        return True, f"Promovido por PR-AUC. ΔPR-AUC={d_pr:+.4f}"

    if d_f1 >= min_delta_f1 and d_prec >= -max_precision_drop and d_rec >= -max_recall_drop:
        return True, f"Promovido por F1. ΔF1={d_f1:+.4f}"

    return False, (
        f"Rechazado. ΔPR-AUC={d_pr:+.4f}, ΔF1={d_f1:+.4f}, "
        f"ΔPrecision={d_prec:+.4f}, ΔRecall={d_rec:+.4f}"
    )


def build_retraining_pairs(
    pares_incrementales: pd.DataFrame,
    pares_historicos_path: Path,
    replay_ratio: float = 0.50,
    random_state: int = 42,
) -> pd.DataFrame:
    if not pares_historicos_path.exists():
        return pares_incrementales.copy()

    pares_hist = pd.read_csv(pares_historicos_path, sep=";", encoding="utf-8-sig")

    n_replay = min(int(len(pares_incrementales) * replay_ratio), len(pares_hist))
    if n_replay <= 0:
        return pares_incrementales.copy()

    replay = pares_hist.sample(
        n=n_replay,
        random_state=random_state,
        replace=False,
    ).copy()

    pares_mix = pd.concat([pares_incrementales, replay], ignore_index=True)
    pares_mix = pares_mix.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return pares_mix


def main() -> None:
    init_seeds()
    ensure_project_dirs()

    maestro_path = require_file(dataset_path(ARCHIVO_MAESTRO), "dataset maestro")
    incremental_path = require_file(
        dataset_path(ARCHIVO_HISTORIAL_INCREMENTAL),
        "dataset historial_facturas_incremental",
    )

    ruta_modelo = require_file(model_path(MODELO_NOMBRE), "modelo base")
    maestro = pd.read_csv(maestro_path, encoding="utf-8-sig")
    historial_incremental = pd.read_csv(incremental_path, encoding="utf-8-sig")

    pares_incrementales = construir_dataset_entrenamiento(
        maestro=maestro,
        historial_facturas=historial_incremental,
        n_neg_por_pos=5,
    )

    pares_incrementales_path = dataset_path(ARCHIVO_PARES_REENTRENAMIENTO)
    pares_incrementales.to_csv(
        pares_incrementales_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
    )

    pares = build_retraining_pairs(
        pares_incrementales=pares_incrementales,
        pares_historicos_path=dataset_path(ARCHIVO_PARES_HISTORICOS),
        replay_ratio=0.50,
        random_state=42,
    )

    print("Pares incrementales:", pares_incrementales.shape)
    print("Pares usados en reentrenamiento:", pares.shape)
    print(pares["label"].value_counts(dropna=False))

    champion = ModeloMatchProducto.cargar(ruta_modelo)
    train_df, valid_df = champion.split_train_valid(pares, test_size=0.2, random_state=42)

    champion_metrics = champion.evaluate_pairs(valid_df)
    print("\nChampion metrics:")
    print(champion_metrics)

    challenger = ModeloMatchProducto.cargar(ruta_modelo)
    fit_report = challenger.fit_incremental_on_split(
        train_df=train_df,
        valid_df=valid_df,
        epochs=6,
        batch_size=256,
        recalcular_threshold=True,
    )

    challenger_metrics = challenger.evaluate_pairs(valid_df)
    print("\nChallenger metrics:")
    print(challenger_metrics)

    promote, reason = should_promote(champion_metrics, challenger_metrics)
    print("\nDecisión:", reason)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": timestamp,
        "model_name": MODELO_NOMBRE,
        "decision": "PROMOTED" if promote else "REJECTED",
        "reason": reason,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "pairs_incrementales_rows": int(len(pares_incrementales)),
        "pairs_retraining_rows": int(len(pares)),
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "fit_report": fit_report,
    }

    reports_dir = ruta_modelo / "retraining_reports"
    save_json(reports_dir / f"retraining_{timestamp}.json", report)

    if not promote:
        print("\nEl modelo actual se mantiene. No se reemplazó nada.")
        return

    backup_path = crear_backup_modelo(ruta_modelo)
    print(f"\nBackup creado en: {backup_path}")

    with tempfile.TemporaryDirectory(prefix="challenger_model_") as tmpdir:
        tmp_model_dir = Path(tmpdir) / MODELO_NOMBRE
        challenger.guardar(tmp_model_dir)

        old_dir = ruta_modelo.parent / f"{ruta_modelo.name}_old_{timestamp}"
        ruta_modelo.rename(old_dir)
        shutil.copytree(tmp_model_dir, ruta_modelo)
        shutil.rmtree(old_dir, ignore_errors=True)

    print("\n--- REENTRENAMIENTO FINALIZADO ---")
    print(f"Modelo promovido en: '{ruta_modelo}'")
    print(f"Nuevo best_threshold: {challenger.best_threshold:.4f}")


if __name__ == "__main__":
    main()
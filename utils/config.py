from __future__ import annotations
import numpy as np
import tensorflow as tf


from pathlib import Path
from typing import Union

SEED = 42
def init_seeds() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTADOS_DIR = PROJECT_ROOT / "resultados"


def ensure_project_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)


def dataset_path(nombre: PathLike) -> Path:
    return DATASETS_DIR / Path(nombre)


def model_path(nombre: PathLike) -> Path:
    return MODELS_DIR / Path(nombre)


def result_path(nombre: PathLike) -> Path:
    return RESULTADOS_DIR / Path(nombre)


def require_file(path: PathLike, descripcion: str = "archivo") -> Path:
    ruta = Path(path)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el {descripcion}: '{ruta}'")
    return ruta
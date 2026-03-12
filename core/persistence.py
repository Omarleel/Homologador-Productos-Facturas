from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf

from . import ModelConfig

PathLike = Union[str, Path]
JsonPayload = Union[dict, list]


class ModelPersistence:
    @staticmethod
    def save(instance: "ModeloMatchProducto", carpeta_modelo: PathLike) -> None:
        if instance.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        path = Path(carpeta_modelo).expanduser()
        path.mkdir(parents=True, exist_ok=True)

        instance.model.save_weights(path / "pesos_modelo.weights.h5")

        vocab_text = instance.assets.text_vec.get_vocabulary()
        ModelPersistence._write_json(path / "text_vocabulary.json", vocab_text)

        text_weights = instance.assets.text_vec.get_weights()
        text_idf = (
            np.array(text_weights[0], dtype=np.float32)
            if text_weights
            else np.array([], dtype=np.float32)
        )
        np.save(path / "text_idf_weights.npy", text_idf, allow_pickle=True)

        ModelPersistence._write_json(
            path / "unit_vocabulary.json",
            instance.assets.unit_lookup.get_vocabulary(),
        )
        ModelPersistence._write_json(
            path / "type_vocabulary.json",
            instance.assets.type_lookup.get_vocabulary(),
        )

        meta = asdict(instance.config) | {"best_threshold": instance.best_threshold}
        ModelPersistence._write_json(path / "meta.json", meta)

        print(f"Modelo y activos guardados exitosamente en: {path}")

    @staticmethod
    def load(instance: "ModeloMatchProducto", carpeta_modelo: PathLike) -> None:
        path = Path(carpeta_modelo).expanduser()

        ModelPersistence._require_file(path / "meta.json")
        ModelPersistence._require_file(path / "pesos_modelo.weights.h5")
        ModelPersistence._require_file(path / "text_vocabulary.json")
        ModelPersistence._require_file(path / "text_idf_weights.npy")
        ModelPersistence._require_file(path / "unit_vocabulary.json")
        ModelPersistence._require_file(path / "type_vocabulary.json")

        vocab_text = ModelPersistence._read_json(path / "text_vocabulary.json")
        text_idf = np.load(path / "text_idf_weights.npy", allow_pickle=True)
        if len(text_idf) == 0:
            text_idf = np.ones(len(vocab_text), dtype=np.float32)

        ModelPersistence._safe_set_text_vocabulary(
            instance.assets.text_vec,
            vocab_text,
            text_idf,
        )

        vocab_unit = ModelPersistence._read_json(path / "unit_vocabulary.json")
        ModelPersistence._safe_set_lookup_vocabulary(
            instance.assets.unit_lookup,
            vocab_unit,
        )

        vocab_type = ModelPersistence._read_json(path / "type_vocabulary.json")
        ModelPersistence._safe_set_lookup_vocabulary(
            instance.assets.type_lookup,
            vocab_type,
        )

        instance.construir()
        instance.model.load_weights(path / "pesos_modelo.weights.h5")

        meta = ModelPersistence._read_json(path / "meta.json")
        instance.best_threshold = float(meta.get("best_threshold", 0.78))

    @staticmethod
    def read_config(carpeta_modelo: PathLike) -> ModelConfig:
        meta_path = Path(carpeta_modelo).expanduser() / "meta.json"
        ModelPersistence._require_file(meta_path)

        meta = ModelPersistence._read_json(meta_path)
        return ModelConfig(
            max_tokens=meta.get("max_tokens", 12000),
            text_embedding_dim=meta.get("text_embedding_dim", 48),
            unit_embedding_dim=meta.get("unit_embedding_dim", 8),
            type_embedding_dim=meta.get("type_embedding_dim", 4),
            dropout_rate=meta.get("dropout_rate", 0.20),
            l2_reg=meta.get("l2_reg", 1e-4),
            learning_rate=meta.get("learning_rate", 8e-4),
        )

    @staticmethod
    def _safe_set_text_vocabulary(
        text_vec: tf.keras.layers.TextVectorization,
        vocabulary: list[str],
        idf_weights: np.ndarray,
    ) -> None:
        try:
            text_vec.set_vocabulary(vocabulary, idf_weights=idf_weights.tolist())
        except Exception:
            text_vec.set_vocabulary(
                vocabulary[1:],
                idf_weights=idf_weights[1:].tolist(),
            )

    @staticmethod
    def _safe_set_lookup_vocabulary(
        lookup: tf.keras.layers.StringLookup,
        vocabulary: list[str],
    ) -> None:
        try:
            lookup.set_vocabulary(vocabulary)
        except Exception:
            lookup.set_vocabulary(vocabulary[1:])

    @staticmethod
    def _write_json(path: Path, payload: JsonPayload) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_json(path: Path) -> JsonPayload:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo requerido: '{path}'")
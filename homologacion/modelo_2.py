import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

class ModeloMatchCodProducto:
    def __init__(
        self,
        max_tokens: int = 12000,
        text_embedding_dim: int = 48,
        unit_embedding_dim: int = 8,
        type_embedding_dim: int = 4,
        dropout_rate: float = 0.20,
        l2_reg: float = 1e-4,
        learning_rate: float = 8e-4,
    ):
        self.max_tokens = max_tokens
        self.text_embedding_dim = text_embedding_dim
        self.unit_embedding_dim = unit_embedding_dim
        self.type_embedding_dim = type_embedding_dim
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate

        self.text_vec = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="tf_idf",
            ngrams=2,
            standardize="lower_and_strip_punctuation",
        )

        self.unit_lookup = tf.keras.layers.StringLookup(
            output_mode="int",
            mask_token=None,
        )
        self.type_lookup = tf.keras.layers.StringLookup(
            output_mode="int",
            mask_token=None,
        )

        self.cost_normalizer = None
        self.peso_normalizer = None
        self.factor_normalizer = None
        self.content_normalizer = None
        self.total_normalizer = None

        self.model: Optional[tf.keras.Model] = None
        self.best_threshold: float = 0.78

    def guardar(self, carpeta_modelo: str) -> None:
        if not os.path.exists(carpeta_modelo):
            os.makedirs(carpeta_modelo)

        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        self.model.save_weights(f"{carpeta_modelo}/pesos_modelo.weights.h5")

        vocab_text = self.text_vec.get_vocabulary()
        with open(f"{carpeta_modelo}/text_vocabulary.json", "w", encoding="utf-8") as f:
            json.dump(vocab_text, f, ensure_ascii=False, indent=2)

        text_weights = self.text_vec.get_weights()
        if len(text_weights) > 0:
            np.save(
                f"{carpeta_modelo}/text_idf_weights.npy",
                np.array(text_weights[0], dtype=np.float32),
                allow_pickle=True,
            )
        else:
            np.save(
                f"{carpeta_modelo}/text_idf_weights.npy",
                np.array([], dtype=np.float32),
                allow_pickle=True,
            )

        with open(f"{carpeta_modelo}/unit_vocabulary.json", "w", encoding="utf-8") as f:
            json.dump(self.unit_lookup.get_vocabulary(), f, ensure_ascii=False, indent=2)

        with open(f"{carpeta_modelo}/type_vocabulary.json", "w", encoding="utf-8") as f:
            json.dump(self.type_lookup.get_vocabulary(), f, ensure_ascii=False, indent=2)

        meta = {
            "max_tokens": self.max_tokens,
            "text_embedding_dim": self.text_embedding_dim,
            "unit_embedding_dim": self.unit_embedding_dim,
            "type_embedding_dim": self.type_embedding_dim,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "learning_rate": self.learning_rate,
            "best_threshold": self.best_threshold,
        }
        with open(f"{carpeta_modelo}/meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Modelo y activos guardados exitosamente en: {carpeta_modelo}")

    @classmethod
    def cargar(cls, carpeta_modelo: str):
        with open(f"{carpeta_modelo}/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        instancia = cls(
            max_tokens=meta.get("max_tokens", 12000),
            text_embedding_dim=meta.get("text_embedding_dim", 48),
            unit_embedding_dim=meta.get("unit_embedding_dim", 8),
            type_embedding_dim=meta.get("type_embedding_dim", 4),
            dropout_rate=meta.get("dropout_rate", 0.20),
            l2_reg=meta.get("l2_reg", 1e-4),
            learning_rate=meta.get("learning_rate", 8e-4),
        )

        with open(f"{carpeta_modelo}/text_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_text = json.load(f)

        text_idf = np.load(f"{carpeta_modelo}/text_idf_weights.npy", allow_pickle=True)
        if len(text_idf) == 0:
            text_idf = np.ones(len(vocab_text), dtype=np.float32)

        try:
            instancia.text_vec.set_vocabulary(vocab_text, idf_weights=text_idf.tolist())
        except Exception:
            instancia.text_vec.set_vocabulary(vocab_text[1:], idf_weights=text_idf[1:].tolist())

        with open(f"{carpeta_modelo}/unit_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_unit = json.load(f)
        try:
            instancia.unit_lookup.set_vocabulary(vocab_unit)
        except Exception:
            instancia.unit_lookup.set_vocabulary(vocab_unit[1:])

        with open(f"{carpeta_modelo}/type_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_type = json.load(f)
        try:
            instancia.type_lookup.set_vocabulary(vocab_type)
        except Exception:
            instancia.type_lookup.set_vocabulary(vocab_type[1:])

        instancia.construir()
        instancia.model.load_weights(f"{carpeta_modelo}/pesos_modelo.weights.h5")
        instancia.best_threshold = float(meta.get("best_threshold", 0.78))
        return instancia

    def adaptar_vocabularios(self, pares: pd.DataFrame) -> None:
        textos = pd.concat(
            [
                pares["fact_text"],
                pares["master_text"],
                pares["fact_base_text"],
                pares["master_base_text"],
            ],
            ignore_index=True,
        ).astype(str).values

        unidades = pd.concat(
            [pares["fact_unit"], pares["master_unit"]],
            ignore_index=True,
        ).astype(str).values

        tipos = pd.concat(
            [pares["fact_type"], pares["master_type"]],
            ignore_index=True,
        ).astype(str).values

        self.text_vec.adapt(textos)
        self.unit_lookup.adapt(unidades)
        self.type_lookup.adapt(tipos)

    def _text_encoder(self) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
        x = self.text_vec(inp)
        x = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(
            self.text_embedding_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.Model(inp, x, name="text_encoder")

    def _categorical_encoder(
        self,
        lookup_layer: tf.keras.layers.StringLookup,
        emb_dim: int,
        name: str,
    ) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(1,), dtype=tf.string, name=f"{name}_input")
        idx = lookup_layer(inp)
        x = tf.keras.layers.Embedding(
            input_dim=max(int(lookup_layer.vocabulary_size()), 2),
            output_dim=emb_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name=f"{name}_embedding",
        )(idx)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            emb_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name=f"{name}_dense",
        )(x)
        return tf.keras.Model(inp, x, name=f"{name}_encoder")

    def construir(self) -> tf.keras.Model:
        fact_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_text")
        master_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_text")
        fact_base_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_base_text")
        master_base_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_base_text")

        fact_unit = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_unit")
        master_unit = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_unit")
        fact_type = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_type")
        master_type = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_type")

        fact_cost = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_cost")
        master_cost = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_cost")
        fact_peso = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_peso")
        master_peso = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_peso")
        fact_factor = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_factor")
        master_factor = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_factor")
        fact_content = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_content")
        master_content = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_content")
        fact_total = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_total")
        master_total = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_total")

        self.cost_normalizer = tf.keras.layers.Normalization(axis=-1, name="cost_normalizer")
        self.peso_normalizer = tf.keras.layers.Normalization(axis=-1, name="peso_normalizer")
        self.factor_normalizer = tf.keras.layers.Normalization(axis=-1, name="factor_normalizer")
        self.content_normalizer = tf.keras.layers.Normalization(axis=-1, name="content_normalizer")
        self.total_normalizer = tf.keras.layers.Normalization(axis=-1, name="total_normalizer")

        fact_cost_norm = self.cost_normalizer(fact_cost)
        master_cost_norm = self.cost_normalizer(master_cost)

        fact_peso_norm = self.peso_normalizer(fact_peso)
        master_peso_norm = self.peso_normalizer(master_peso)

        fact_factor_norm = self.factor_normalizer(fact_factor)
        master_factor_norm = self.factor_normalizer(master_factor)

        fact_content_norm = self.content_normalizer(fact_content)
        master_content_norm = self.content_normalizer(master_content)

        fact_total_norm = self.total_normalizer(fact_total)
        master_total_norm = self.total_normalizer(master_total)

        text_encoder = self._text_encoder()
        unit_encoder = self._categorical_encoder(
            lookup_layer=self.unit_lookup,
            emb_dim=self.unit_embedding_dim,
            name="unit",
        )
        type_encoder = self._categorical_encoder(
            lookup_layer=self.type_lookup,
            emb_dim=self.type_embedding_dim,
            name="type",
        )

        fact_text_vec = text_encoder(fact_text)
        master_text_vec = text_encoder(master_text)
        fact_base_vec = text_encoder(fact_base_text)
        master_base_vec = text_encoder(master_base_text)

        fact_unit_vec = unit_encoder(fact_unit)
        master_unit_vec = unit_encoder(master_unit)

        fact_type_vec = type_encoder(fact_type)
        master_type_vec = type_encoder(master_type)

        diff_text = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="diff_text",
        )([fact_text_vec, master_text_vec])

        diff_base = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="diff_base",
        )([fact_base_vec, master_base_vec])

        diff_unit = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="diff_unit",
        )([fact_unit_vec, master_unit_vec])

        diff_type = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="diff_type",
        )([fact_type_vec, master_type_vec])

        text_cos = tf.keras.layers.Dot(
            axes=1,
            normalize=True,
            name="text_cos",
        )([fact_text_vec, master_text_vec])

        base_cos = tf.keras.layers.Dot(
            axes=1,
            normalize=True,
            name="base_cos",
        )([fact_base_vec, master_base_vec])

        unit_cos = tf.keras.layers.Dot(
            axes=1,
            normalize=True,
            name="unit_cos",
        )([fact_unit_vec, master_unit_vec])

        type_cos = tf.keras.layers.Dot(
            axes=1,
            normalize=True,
            name="type_cos",
        )([fact_type_vec, master_type_vec])

        fact_unit_idx = self.unit_lookup(fact_unit)
        master_unit_idx = self.unit_lookup(master_unit)
        unit_match = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32),
            name="unit_match",
        )([fact_unit_idx, master_unit_idx])

        fact_type_idx = self.type_lookup(fact_type)
        master_type_idx = self.type_lookup(master_type)
        type_match = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32),
            name="type_match",
        )([fact_type_idx, master_type_idx])


        def scalar_float_lambda(fn, name: str):
            return tf.keras.layers.Lambda(
                fn,
                output_shape=(1,),
                dtype=tf.float32,
                name=name,
            )

        fact_type_known = scalar_float_lambda(
            lambda x: tf.cast(
                tf.not_equal(x, tf.constant("NONE", dtype=tf.string)),
                tf.float32,
            ),
            "fact_type_known",
        )(fact_type)

        master_type_known = scalar_float_lambda(
            lambda x: tf.cast(
                tf.not_equal(x, tf.constant("NONE", dtype=tf.string)),
                tf.float32,
            ),
            "master_type_known",
        )(master_type)

        type_available = tf.keras.layers.Multiply(name="type_available")(
            [fact_type_known, master_type_known]
        )

        content_available = scalar_float_lambda(
            lambda x: tf.cast(
                tf.logical_and(x[0] > 0.0, x[1] > 0.0),
                tf.float32,
            ),
            "content_available",
        )([fact_content, master_content])

        total_available = scalar_float_lambda(
            lambda x: tf.cast(
                tf.logical_and(x[0] > 0.0, x[1] > 0.0),
                tf.float32,
            ),
            "total_available",
        )([fact_total, master_total])

        peso_available = scalar_float_lambda(
            lambda x: tf.cast(
                tf.logical_and(x[0] > 0.0, x[1] > 0.0),
                tf.float32,
            ),
            "peso_available",
        )([fact_peso, master_peso])

        cost_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="cost_diff",
        )([fact_cost_norm, master_cost_norm])

        peso_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="peso_diff",
        )([fact_peso_norm, master_peso_norm])

        factor_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="factor_diff",
        )([fact_factor_norm, master_factor_norm])

        content_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="content_diff",
        )([fact_content_norm, master_content_norm])

        total_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name="total_diff",
        )([fact_total_norm, master_total_norm])

        cost_sim = tf.keras.layers.Lambda(
            lambda x: tf.exp(-1.2 * tf.abs(x[0] - x[1])),
            name="cost_sim",
        )([fact_cost_norm, master_cost_norm])

        peso_sim = tf.keras.layers.Lambda(
            lambda x: tf.exp(-1.8 * tf.abs(x[0] - x[1])),
            name="peso_sim",
        )([fact_peso_norm, master_peso_norm])

        factor_sim = tf.keras.layers.Lambda(
            lambda x: tf.exp(-2.0 * tf.abs(x[0] - x[1])),
            name="factor_sim",
        )([fact_factor_norm, master_factor_norm])

        content_sim = tf.keras.layers.Lambda(
            lambda x: tf.exp(-2.0 * tf.abs(x[0] - x[1])),
            name="content_sim",
        )([fact_content_norm, master_content_norm])

        total_sim = tf.keras.layers.Lambda(
            lambda x: tf.exp(-2.2 * tf.abs(x[0] - x[1])),
            name="total_sim",
        )([fact_total_norm, master_total_norm])

        factor_close = scalar_float_lambda(
            lambda x: tf.cast(tf.abs(x[0] - x[1]) <= 0.10, tf.float32),
            "factor_close",
        )([fact_factor, master_factor])

        content_close = scalar_float_lambda(
            lambda x: tf.cast(tf.abs(x[0] - x[1]) <= 0.12, tf.float32),
            "content_close",
        )([fact_content, master_content])

        total_close = scalar_float_lambda(
            lambda x: tf.cast(tf.abs(x[0] - x[1]) <= 0.12, tf.float32),
            "total_close",
        )([fact_total, master_total])

        peso_rel_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]) / tf.maximum(
                tf.maximum(tf.abs(x[0]), tf.abs(x[1])),
                1e-6,
            ),
            name="peso_rel_diff",
        )([fact_peso, master_peso])

        peso_close = scalar_float_lambda(
            lambda x: tf.cast(x <= 0.08, tf.float32),
            "peso_close",
        )(peso_rel_diff)

        lexical_features = tf.keras.layers.Concatenate(name="lexical_features")([
            fact_text_vec,
            master_text_vec,
            diff_text,
            text_cos,
            fact_base_vec,
            master_base_vec,
            diff_base,
            base_cos,
            fact_unit_vec,
            master_unit_vec,
            diff_unit,
            unit_cos,
            unit_match,
            fact_cost_norm,
            master_cost_norm,
            cost_diff,
            cost_sim,
        ])

        lexical_repr = tf.keras.layers.BatchNormalization()(lexical_features)
        lexical_repr = tf.keras.layers.Dense(
            192,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="lexical_dense_1",
        )(lexical_repr)
        lexical_repr = tf.keras.layers.Dropout(self.dropout_rate)(lexical_repr)
        lexical_repr = tf.keras.layers.Dense(
            96,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="lexical_dense_2",
        )(lexical_repr)

        presentation_features = tf.keras.layers.Concatenate(name="presentation_features")([
            fact_type_vec,
            master_type_vec,
            diff_type,
            type_cos,
            type_match,
            type_available,
            fact_peso_norm,
            master_peso_norm,
            peso_diff,
            peso_sim,
            peso_close,
            fact_factor_norm,
            master_factor_norm,
            factor_diff,
            factor_sim,
            factor_close,
            fact_content_norm,
            master_content_norm,
            content_diff,
            content_sim,
            content_close,
            content_available,
            fact_total_norm,
            master_total_norm,
            total_diff,
            total_sim,
            total_close,
            total_available,
        ])

        presentation_repr = tf.keras.layers.BatchNormalization()(presentation_features)
        presentation_repr = tf.keras.layers.Dense(
            160,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="presentation_dense_1",
        )(presentation_repr)
        presentation_repr = tf.keras.layers.Dropout(self.dropout_rate)(presentation_repr)
        presentation_repr = tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="presentation_dense_2",
        )(presentation_repr)

        fusion = tf.keras.layers.Concatenate(name="fusion_features")([
            lexical_repr,
            presentation_repr,
            text_cos,
            base_cos,
            unit_cos,
            type_cos,
            unit_match,
            type_match,
            cost_sim,
            peso_sim,
            factor_sim,
            content_sim,
            total_sim,
            factor_close,
            content_close,
            total_close,
            peso_close,
        ])

        x = tf.keras.layers.BatchNormalization()(fusion)
        x = tf.keras.layers.Dense(
            192,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="fusion_dense_1",
        )(x)
        x = tf.keras.layers.Dropout(self.dropout_rate + 0.05)(x)
        x = tf.keras.layers.Dense(
            96,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="fusion_dense_2",
        )(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        pre_prob = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="pre_prob",
        )(x)

        type_gate_base = scalar_float_lambda(
            lambda z: 0.08 + 0.92 * z,
            "type_gate_base",
        )(type_match)

        type_gate = scalar_float_lambda(
            lambda z: (z[0] * z[1]) + ((1.0 - z[0]) * 1.0),
            "type_gate",
        )([type_available, type_gate_base])

        factor_gate = scalar_float_lambda(
            lambda z: 0.02 + 0.98 * (0.80 * z[0] + 0.20 * z[1]),
            "factor_gate",
        )([factor_sim, factor_close])

        content_gate_base = scalar_float_lambda(
            lambda z: 0.05 + 0.95 * (0.65 * z[0] + 0.25 * z[1] + 0.10 * z[2]),
            "content_gate_base",
        )([content_sim, content_close, type_match])

        content_gate = scalar_float_lambda(
            lambda z: (z[0] * z[1]) + ((1.0 - z[0]) * 1.0),
            "content_gate",
        )([content_available, content_gate_base])

        total_gate_base = scalar_float_lambda(
            lambda z: 0.02 + 0.98 * (0.70 * z[0] + 0.20 * z[1] + 0.10 * z[2]),
            "total_gate_base",
        )([total_sim, total_close, type_match])

        total_gate = scalar_float_lambda(
            lambda z: (z[0] * z[1]) + ((1.0 - z[0]) * 1.0),
            "total_gate",
        )([total_available, total_gate_base])

        peso_gate_base = scalar_float_lambda(
            lambda z: 0.10 + 0.90 * (0.70 * z[0] + 0.30 * z[1]),
            "peso_gate_base",
        )([peso_sim, peso_close])

        peso_gate = scalar_float_lambda(
            lambda z: (z[0] * z[1]) + ((1.0 - z[0]) * 1.0),
            "peso_gate",
        )([peso_available, peso_gate_base])

        gate_mean = scalar_float_lambda(
            lambda z: (
                0.20 * z[0]
                + 0.28 * z[1]
                + 0.18 * z[2]
                + 0.22 * z[3]
                + 0.12 * z[4]
            ),
            "presentation_gate_mean",
        )([type_gate, factor_gate, content_gate, total_gate, peso_gate])

        gate_min = scalar_float_lambda(
            lambda z: tf.reduce_min(tf.concat(z, axis=1), axis=1, keepdims=True),
            "presentation_gate_min",
        )([factor_gate, content_gate, total_gate, type_gate])

        presentation_gate = scalar_float_lambda(
            lambda z: 0.70 * z[0] + 0.30 * z[1],
            "presentation_gate",
        )([gate_mean, gate_min])

        fusion2 = tf.keras.layers.Concatenate(name="fusion_with_gate")([
            x,
            presentation_gate,
            gate_mean,
            gate_min,
            type_gate,
            factor_gate,
            content_gate,
            total_gate,
            peso_gate,
        ])

        fusion2 = tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )(fusion2)
        fusion2 = tf.keras.layers.Dropout(self.dropout_rate)(fusion2)

        y = tf.keras.layers.Dense(1, activation="sigmoid", name="match_prob")(fusion2)

        model = tf.keras.Model(
            inputs=[
                fact_text,
                master_text,
                fact_base_text,
                master_base_text,
                fact_unit,
                master_unit,
                fact_type,
                master_type,
                fact_cost,
                master_cost,
                fact_peso,
                master_peso,
                fact_factor,
                master_factor,
                fact_content,
                master_content,
                fact_total,
                master_total,
            ],
            outputs=y,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
            ],
        )

        self.model = model
        return model

    @staticmethod
    def _to_ds(
        df: pd.DataFrame,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: int = 256,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        x = {
            "fact_text": df["fact_text"].astype(str).values.reshape(-1, 1),
            "master_text": df["master_text"].astype(str).values.reshape(-1, 1),
            "fact_base_text": df["fact_base_text"].astype(str).values.reshape(-1, 1),
            "master_base_text": df["master_base_text"].astype(str).values.reshape(-1, 1),
            "fact_unit": df["fact_unit"].astype(str).values.reshape(-1, 1),
            "master_unit": df["master_unit"].astype(str).values.reshape(-1, 1),
            "fact_type": df["fact_type"].astype(str).values.reshape(-1, 1),
            "master_type": df["master_type"].astype(str).values.reshape(-1, 1),
            "fact_cost": df["fact_cost"].astype(np.float32).values.reshape(-1, 1),
            "master_cost": df["master_cost"].astype(np.float32).values.reshape(-1, 1),
            "fact_peso": df["fact_peso"].astype(np.float32).values.reshape(-1, 1),
            "master_peso": df["master_peso"].astype(np.float32).values.reshape(-1, 1),
            "fact_factor": df["fact_factor"].astype(np.float32).values.reshape(-1, 1),
            "master_factor": df["master_factor"].astype(np.float32).values.reshape(-1, 1),
            "fact_content": df["fact_content"].astype(np.float32).values.reshape(-1, 1),
            "master_content": df["master_content"].astype(np.float32).values.reshape(-1, 1),
            "fact_total": df["fact_total"].astype(np.float32).values.reshape(-1, 1),
            "master_total": df["master_total"].astype(np.float32).values.reshape(-1, 1),
        }
        y = df["label"].astype(np.float32).values

        if sample_weight is not None:
            ds = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle:
            ds = ds.shuffle(min(len(df), 10000), seed=42)

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(self, pares: pd.DataFrame, epochs: int = 12, batch_size: int = 256) -> None:
        groups = (
            pares["RucProveedor"].astype(str).fillna("") + "|" +
            pares["fact_cod"].astype(str).fillna("") + "|" +
            pares["fact_text"].astype(str).fillna("")
        )

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(gss.split(pares, pares["label"], groups=groups))

        train_df = pares.iloc[train_idx].copy().reset_index(drop=True)
        valid_df = pares.iloc[valid_idx].copy().reset_index(drop=True)

        self.adaptar_vocabularios(train_df)

        if self.model is None:
            self.construir()

        self.cost_normalizer.adapt(
            np.concatenate([train_df["fact_cost"].values, train_df["master_cost"].values]).reshape(-1, 1)
        )
        self.peso_normalizer.adapt(
            np.concatenate([train_df["fact_peso"].values, train_df["master_peso"].values]).reshape(-1, 1)
        )
        self.factor_normalizer.adapt(
            np.concatenate([train_df["fact_factor"].values, train_df["master_factor"].values]).reshape(-1, 1)
        )
        self.content_normalizer.adapt(
            np.concatenate([train_df["fact_content"].values, train_df["master_content"].values]).reshape(-1, 1)
        )
        self.total_normalizer.adapt(
            np.concatenate([train_df["fact_total"].values, train_df["master_total"].values]).reshape(-1, 1)
        )

        n_pos = max(int((train_df["label"] == 1).sum()), 1)
        n_neg = max(int((train_df["label"] == 0).sum()), 1)

        sample_weight = np.ones(len(train_df), dtype=np.float32)
        sample_weight[train_df["label"].values == 1] = n_neg / n_pos

        neg_mask = train_df["label"].values == 0

        cost_gap = np.abs(train_df["fact_cost"].values - train_df["master_cost"].values)
        factor_gap = np.abs(train_df["fact_factor"].values - train_df["master_factor"].values)
        content_gap = np.abs(train_df["fact_content"].values - train_df["master_content"].values)
        total_gap = np.abs(train_df["fact_total"].values - train_df["master_total"].values)

        peso_a = train_df["fact_peso"].values.astype(np.float32)
        peso_b = train_df["master_peso"].values.astype(np.float32)
        peso_rel_gap = np.abs(peso_a - peso_b) / np.maximum(np.maximum(np.abs(peso_a), np.abs(peso_b)), 1e-6)

        same_type = (train_df["fact_type"].astype(str).values == train_df["master_type"].astype(str).values).astype(np.float32)
        known_type = (
            (train_df["fact_type"].astype(str).values != "NONE") &
            (train_df["master_type"].astype(str).values != "NONE")
        ).astype(np.float32)

        same_base = (
            train_df["fact_base_text"].astype(str).str.strip().values
            == train_df["master_base_text"].astype(str).str.strip().values
        ).astype(np.float32)

        dificultad_neg = (
            0.18 * np.exp(-1.2 * cost_gap)
            + 0.18 * np.exp(-2.0 * factor_gap)
            + 0.16 * np.exp(-2.0 * content_gap)
            + 0.26 * np.exp(-2.2 * total_gap)
            + 0.12 * np.exp(-4.0 * peso_rel_gap)
            + 0.10 * (same_type * known_type)
        )

        conflicto_presentacion = (
            0.30 * (factor_gap > 0.10).astype(np.float32)
            + 0.20 * (content_gap > 0.12).astype(np.float32)
            + 0.30 * (total_gap > 0.12).astype(np.float32)
            + 0.20 * ((same_type == 0).astype(np.float32) * known_type)
        )

        bonus_mismo_nombre = 1.0 + 0.60 * same_base

        sample_weight[neg_mask] *= (
            1.0
            + (2.4 * dificultad_neg[neg_mask] * bonus_mismo_nombre[neg_mask])
            + (1.6 * conflicto_presentacion[neg_mask] * bonus_mismo_nombre[neg_mask])
        )

        sample_weight = np.clip(sample_weight, 1.0, 6.0)

        ds_train = self._to_ds(train_df, sample_weight=sample_weight, batch_size=batch_size, shuffle=True)
        ds_valid = self._to_ds(valid_df, batch_size=batch_size, shuffle=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc",
                mode="max",
                factor=0.5,
                patience=1,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        self.model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        y_valid = valid_df["label"].astype(int).values
        p_valid = self.model.predict(ds_valid, verbose=0).reshape(-1)

        best_th = 0.50
        best_score = -1.0
        best_precision = -1.0
        best_recall = -1.0

        for th in np.arange(0.30, 0.98, 0.01):
            y_hat = (p_valid >= th).astype(int)
            prec = precision_score(y_valid, y_hat, zero_division=0)
            rec = recall_score(y_valid, y_hat, zero_division=0)
            f1 = fbeta_score(y_valid, y_hat, beta=1.0, zero_division=0)

            if (f1 > best_score) or (f1 == best_score and prec > best_precision):
                best_score = float(f1)
                best_precision = float(prec)
                best_recall = float(rec)
                best_th = float(th)

        self.best_threshold = best_th
        print(
            f"Mejor umbral validación: {self.best_threshold:.2f} | "
            f"F1={best_score:.4f} | precision={best_precision:.4f} | recall={best_recall:.4f}"
        )
    def predict_pairs(self, pares: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de predecir.")

        ds = self._to_ds(pares.assign(label=0), batch_size=batch_size, shuffle=False)
        preds = self.model.predict(ds, verbose=0).reshape(-1)
        return preds
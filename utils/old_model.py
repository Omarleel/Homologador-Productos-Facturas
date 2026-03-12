import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class ModeloMatchCodProducto:
    def __init__(
        self,
        max_tokens: int = 12000,
        text_embedding_dim: int = 32,
    ):
        self.max_tokens = max_tokens
        self.text_embedding_dim = text_embedding_dim

        self.text_vec = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="tf_idf",
            ngrams=2,
            standardize="lower_and_strip_punctuation",
        )

        self.unit_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)
        self.type_lookup = tf.keras.layers.StringLookup(output_mode="int", mask_token=None)

        self.cost_normalizer = None
        self.peso_normalizer = None
        self.factor_normalizer = None
        self.content_normalizer = None
        self.total_normalizer = None

        self.model: Optional[tf.keras.Model] = None
        self.best_threshold: float = 0.72

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
            max_tokens=meta["max_tokens"],
            text_embedding_dim=meta["text_embedding_dim"],
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
        instancia.best_threshold = float(meta.get("best_threshold", 0.72))
        return instancia

    def adaptar_vocabularios(self, pares: pd.DataFrame) -> None:
        textos = pd.concat(
            [pares["fact_text"], pares["master_text"], pares["fact_base_text"], pares["master_base_text"]],
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
        inp = tf.keras.Input(shape=(1,), dtype=tf.string)
        x = self.text_vec(inp)
        x = tf.keras.layers.Dense(192, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.20)(x)
        x = tf.keras.layers.Dense(self.text_embedding_dim, activation="relu")(x)
        return tf.keras.Model(inp, x, name="text_encoder")

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

        fact_text_vec = text_encoder(fact_text)
        master_text_vec = text_encoder(master_text)
        fact_base_vec = text_encoder(fact_base_text)
        master_base_vec = text_encoder(master_base_text)

        diff_text = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="diff_text")([fact_text_vec, master_text_vec])
        diff_base = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="diff_base")([fact_base_vec, master_base_vec])

        text_cos = tf.keras.layers.Dot(axes=1, normalize=True, name="text_cos")([fact_text_vec, master_text_vec])
        base_cos = tf.keras.layers.Dot(axes=1, normalize=True, name="base_cos")([fact_base_vec, master_base_vec])

        fact_unit_idx = self.unit_lookup(fact_unit)
        master_unit_idx = self.unit_lookup(master_unit)
        unit_match = tf.keras.layers.Lambda(lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32), name="unit_match")([fact_unit_idx, master_unit_idx])

        fact_type_idx = self.type_lookup(fact_type)
        master_type_idx = self.type_lookup(master_type)
        type_match = tf.keras.layers.Lambda(lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32), name="type_match")([fact_type_idx, master_type_idx])

        cost_sim = tf.keras.layers.Lambda(lambda x: tf.exp(-tf.abs(x[0] - x[1])), name="cost_sim")([fact_cost_norm, master_cost_norm])
        peso_sim = tf.keras.layers.Lambda(lambda x: tf.exp(-1.5 * tf.abs(x[0] - x[1])), name="peso_sim")([fact_peso_norm, master_peso_norm])
        factor_sim = tf.keras.layers.Lambda(lambda x: tf.exp(-1.6 * tf.abs(x[0] - x[1])), name="factor_sim")([fact_factor_norm, master_factor_norm])
        content_sim = tf.keras.layers.Lambda(lambda x: tf.exp(-1.6 * tf.abs(x[0] - x[1])), name="content_sim")([fact_content_norm, master_content_norm])
        total_sim = tf.keras.layers.Lambda(lambda x: tf.exp(-1.6 * tf.abs(x[0] - x[1])), name="total_sim")([fact_total_norm, master_total_norm])

        x = tf.keras.layers.Concatenate()([
            fact_text_vec, master_text_vec, diff_text, text_cos,
            fact_base_vec, master_base_vec, diff_base, base_cos,
            unit_match, type_match,
            fact_cost_norm, master_cost_norm, cost_sim,
            fact_peso_norm, master_peso_norm, peso_sim,
            fact_factor_norm, master_factor_norm, factor_sim,
            fact_content_norm, master_content_norm, content_sim,
            fact_total_norm, master_total_norm, total_sim,
        ])

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(192, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(96, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        logits_match = tf.keras.layers.Dense(1, activation="sigmoid", name="pre_prob")(x)

        cost_ratio = tf.keras.layers.Lambda(lambda z: z[0] / (z[1] + 1e-6), name="cost_ratio")([fact_cost_norm, master_cost_norm])

        cost_gate = tf.keras.layers.Lambda(
            lambda z: tf.exp(-4.0 * tf.square(z - 1.0)),
            name="cost_gate",
        )(cost_ratio)

        factor_gate = tf.keras.layers.Lambda(
            lambda z: tf.exp(-1.5 * tf.abs(z[0] - z[1])),
            name="factor_gate",
        )([fact_factor_norm, master_factor_norm])

        total_gate = tf.keras.layers.Lambda(
            lambda z: tf.exp(-1.8 * tf.abs(z[0] - z[1])) * (0.35 + 0.65 * z[2]),
            name="total_gate",
        )([fact_total_norm, master_total_norm, type_match])

        confianza_total = tf.keras.layers.Multiply(name="total_confidence")(
            [logits_match, cost_gate, factor_gate, total_gate]
        )
        y = tf.keras.layers.Activation("linear", name="match_prob")(confianza_total)

        model = tf.keras.Model(
            inputs=[
                fact_text, master_text, fact_base_text, master_base_text,
                fact_unit, master_unit, fact_type, master_type,
                fact_cost, master_cost, fact_peso, master_peso,
                fact_factor, master_factor, fact_content, master_content,
                fact_total, master_total,
            ],
            outputs=y,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
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

    def fit(self, pares: pd.DataFrame, epochs: int = 10, batch_size: int = 256) -> None:
        self.adaptar_vocabularios(pares)

        if self.model is None:
            self.construir()

        self.cost_normalizer.adapt(
            np.concatenate([pares["fact_cost"].values, pares["master_cost"].values]).reshape(-1, 1)
        )
        self.peso_normalizer.adapt(
            np.concatenate([pares["fact_peso"].values, pares["master_peso"].values]).reshape(-1, 1)
        )
        self.factor_normalizer.adapt(
            np.concatenate([pares["fact_factor"].values, pares["master_factor"].values]).reshape(-1, 1)
        )
        self.content_normalizer.adapt(
            np.concatenate([pares["fact_content"].values, pares["master_content"].values]).reshape(-1, 1)
        )
        self.total_normalizer.adapt(
            np.concatenate([pares["fact_total"].values, pares["master_total"].values]).reshape(-1, 1)
        )

        train_df, valid_df = train_test_split(
            pares,
            test_size=0.2,
            random_state=42,
            stratify=pares["label"],
        )

        train_df = train_df.copy()
        n_pos = max(int((train_df["label"] == 1).sum()), 1)
        n_neg = max(int((train_df["label"] == 0).sum()), 1)

        sample_weight = np.ones(len(train_df), dtype=np.float32)
        sample_weight[train_df["label"].values == 1] = n_neg / n_pos

        neg_mask = train_df["label"].values == 0
        dificultad_neg = (
            0.30 * np.exp(-np.abs(train_df["fact_cost"].values - train_df["master_cost"].values))
            + 0.25 * np.exp(-np.abs(train_df["fact_factor"].values - train_df["master_factor"].values))
            + 0.45 * np.exp(-np.abs(train_df["fact_total"].values - train_df["master_total"].values))
        )
        sample_weight[neg_mask] *= (1.0 + 2.5 * dificultad_neg[neg_mask])

        ds_train = self._to_ds(train_df, sample_weight=sample_weight, batch_size=batch_size, shuffle=True)
        ds_valid = self._to_ds(valid_df, batch_size=batch_size, shuffle=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=2,
                restore_best_weights=True,
            )
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

        best_th = 0.72
        best_score = -1.0
        best_precision = -1.0
        best_recall = -1.0

        for th in np.arange(0.40, 0.96, 0.02):
            y_hat = (p_valid >= th).astype(int)
            prec = precision_score(y_valid, y_hat, zero_division=0)
            rec = recall_score(y_valid, y_hat, zero_division=0)
            f05 = fbeta_score(y_valid, y_hat, beta=0.5, zero_division=0)

            if (f05 > best_score) or (f05 == best_score and prec > best_precision):
                best_score = float(f05)
                best_precision = float(prec)
                best_recall = float(rec)
                best_th = float(th)

        self.best_threshold = best_th
        print(
            f"Mejor umbral validación: {self.best_threshold:.2f} | "
            f"F0.5={best_score:.4f} | precision={best_precision:.4f} | recall={best_recall:.4f}"
        )

    def predict_pairs(self, pares: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de predecir.")

        ds = self._to_ds(pares.assign(label=0), batch_size=batch_size, shuffle=False)
        preds = self.model.predict(ds, verbose=0).reshape(-1)
        return preds
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class ModeloMatchCodProducto:
    def __init__(self, max_tokens: int = 6000, emb_dim: int = 48):
        self.max_tokens = max_tokens
        self.emb_dim = emb_dim

        self.text_vec = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="tf_idf",
            ngrams=2,
            standardize="lower_and_strip_punctuation",
        )

        self.unit_lookup = tf.keras.layers.StringLookup(
            output_mode="int",
            mask_token=None
        )

        self.model: Optional[tf.keras.Model] = None
        self.best_threshold: float = 0.65

    def guardar(self, carpeta_modelo: str) -> None:
        if not os.path.exists(carpeta_modelo):
            os.makedirs(carpeta_modelo)

        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        # 1) Guardar pesos de la red neuronal
        self.model.save_weights(f"{carpeta_modelo}/pesos_modelo.weights.h5")

        # 2) Guardar vocabulario de TextVectorization
        vocab_text = self.text_vec.get_vocabulary()
        with open(f"{carpeta_modelo}/text_vocabulary.json", "w", encoding="utf-8") as f:
            json.dump(vocab_text, f, ensure_ascii=False, indent=2)

        # 3) Guardar IDF del tf-idf
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

        # 4) Guardar vocabulario de StringLookup
        vocab_unit = self.unit_lookup.get_vocabulary()
        with open(f"{carpeta_modelo}/unit_vocabulary.json", "w", encoding="utf-8") as f:
            json.dump(vocab_unit, f, ensure_ascii=False, indent=2)

        # 5) Guardar metadatos
        meta = {
            "max_tokens": self.max_tokens,
            "emb_dim": self.emb_dim,
            "best_threshold": self.best_threshold,
        }
        with open(f"{carpeta_modelo}/meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Pesos (incluyendo TF-IDF y vocabularios) guardados en: {carpeta_modelo}")

    @classmethod
    def cargar(cls, carpeta_modelo: str):
        with open(f"{carpeta_modelo}/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        instancia = cls(
            max_tokens=meta["max_tokens"],
            emb_dim=meta["emb_dim"],
        )

        # 1) Restaurar vocabulario de TextVectorization
        with open(f"{carpeta_modelo}/text_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_text = json.load(f)

        text_idf = np.load(
            f"{carpeta_modelo}/text_idf_weights.npy",
            allow_pickle=True
        )

        # Asegurar que idf_weights tenga la longitud adecuada
        if len(text_idf) == 0:
            # Si no hay pesos (por ejemplo, archivo vacío), crear unos (IDF por defecto)
            text_idf = np.ones(len(vocab_text), dtype=np.float32)

        try:
            instancia.text_vec.set_vocabulary(vocab_text, idf_weights=text_idf.tolist())
        except Exception:
            # Fallback: quitar el primer token (reservado) si la versión de Keras lo exige
            instancia.text_vec.set_vocabulary(vocab_text[1:], idf_weights=text_idf[1:].tolist())

        # 2) Restaurar vocabulario de StringLookup
        with open(f"{carpeta_modelo}/unit_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_unit = json.load(f)

        try:
            instancia.unit_lookup.set_vocabulary(vocab_unit)
        except Exception:
            # Algunas versiones esperan vocabulario sin token especial
            instancia.unit_lookup.set_vocabulary(vocab_unit[1:])

        # 3) Construir el modelo ya con preprocessors restaurados
        instancia.construir()

        # 4) Cargar pesos de la red
        instancia.model.load_weights(f"{carpeta_modelo}/pesos_modelo.weights.h5")
        instancia.best_threshold = meta["best_threshold"]

        return instancia

    def adaptar_vocabularios(self, pares: pd.DataFrame) -> None:
        textos = pd.concat(
            [pares["fact_text"], pares["master_text"]],
            ignore_index=True
        ).astype(str).values

        unidades = pd.concat(
            [pares["fact_unit"], pares["master_unit"]],
            ignore_index=True
        ).astype(str).values

        self.text_vec.adapt(textos)
        self.unit_lookup.adapt(unidades)

    def _text_encoder(self) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(1,), dtype=tf.string)
        x = self.text_vec(inp)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.20)(x)
        x = tf.keras.layers.Dense(self.emb_dim, activation="relu")(x)
        return tf.keras.Model(inp, x, name="text_encoder")

    def construir(self) -> tf.keras.Model:
        fact_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_text")
        master_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_text")
        fact_unit = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_unit")
        master_unit = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_unit")
        fact_cost = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_cost")
        master_cost = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_cost")

        text_encoder = self._text_encoder()

        fact_text_vec = text_encoder(fact_text)
        master_text_vec = text_encoder(master_text)

        diff_text = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1])
        )([fact_text_vec, master_text_vec])

        text_cos = tf.keras.layers.Dot(axes=1, normalize=True)(
            [fact_text_vec, master_text_vec]
        )

        fact_unit_idx = self.unit_lookup(fact_unit)
        master_unit_idx = self.unit_lookup(master_unit)

        unit_match = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32)
        )([fact_unit_idx, master_unit_idx])

        cost_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1])
        )([fact_cost, master_cost])

        cost_sim = tf.keras.layers.Lambda(
            lambda x: tf.exp(-x)
        )(cost_diff)

        x = tf.keras.layers.Concatenate()([
            fact_text_vec,
            master_text_vec,
            diff_text,
            text_cos,
            unit_match,
            fact_cost,
            master_cost,
            cost_diff,
            cost_sim,
        ])

        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        y = tf.keras.layers.Dense(1, activation="sigmoid", name="match_prob")(x)

        model = tf.keras.Model(
            inputs=[fact_text, master_text, fact_unit, master_unit, fact_cost, master_cost],
            outputs=y,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
            ],
        )

        self.model = model
        return model

    @staticmethod
    def _to_ds(df: pd.DataFrame, batch_size: int = 256, shuffle: bool = False) -> tf.data.Dataset:
        x = {
            "fact_text": df["fact_text"].astype(str).values.reshape(-1, 1),
            "master_text": df["master_text"].astype(str).values.reshape(-1, 1),
            "fact_unit": df["fact_unit"].astype(str).values.reshape(-1, 1),
            "master_unit": df["master_unit"].astype(str).values.reshape(-1, 1),
            "fact_cost": df["fact_cost"].astype(np.float32).values.reshape(-1, 1),
            "master_cost": df["master_cost"].astype(np.float32).values.reshape(-1, 1),
        }

        y = df["label"].astype(np.float32).values

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(min(len(df), 10000), seed=42)

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(self, pares: pd.DataFrame, epochs: int = 6, batch_size: int = 256) -> None:
        self.adaptar_vocabularios(pares)

        if self.model is None:
            self.construir()

        train_df, valid_df = train_test_split(
            pares,
            test_size=0.2,
            random_state=42,
            stratify=pares["label"],
        )

        ds_train = self._to_ds(train_df, batch_size=batch_size, shuffle=True)
        ds_valid = self._to_ds(valid_df, batch_size=batch_size, shuffle=False)

        n_pos = max(int((train_df["label"] == 1).sum()), 1)
        n_neg = max(int((train_df["label"] == 0).sum()), 1)
        class_weight = {
            0: 1.0,
            1: n_neg / n_pos,
        }

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=1,
                restore_best_weights=True,
            )
        ]

        self.model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        y_valid = valid_df["label"].astype(int).values
        p_valid = self.model.predict(ds_valid, verbose=0).reshape(-1)

        best_th = 0.65
        best_acc = -1.0

        for th in np.arange(0.35, 0.91, 0.02):
            y_hat = (p_valid >= th).astype(int)
            acc = (y_hat == y_valid).mean()
            if acc > best_acc:
                best_acc = acc
                best_th = float(th)

        self.best_threshold = best_th
        print(f"Mejor umbral validación: {self.best_threshold:.2f} | val_acc={best_acc:.4f}")

    def predict_pairs(self, pares: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de predecir.")

        ds = self._to_ds(pares.assign(label=0), batch_size=batch_size, shuffle=False)
        preds = self.model.predict(ds, verbose=0).reshape(-1)
        return preds
import json
import os
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class ModeloMatchCodProducto:
    def __init__(
        self, 
        max_tokens: int = 10000,
        text_embedding_dim: int = 16,  # Reducido de 48/24 a 16
        ):
        self.max_tokens = max_tokens
        self.text_embedding_dim = text_embedding_dim
        self.cost_normalizer = None
        self.peso_normalizer = None
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

        # 1) Guardar pesos en formato nativo de TensorFlow (.weights.tf)
        # Esto soluciona el NotImplementedError con StringLookup y TextVectorization
        self.model.save_weights(f"{carpeta_modelo}/pesos_modelo.weights.tf")

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

        # 1) Restaurar vocabulario de TextVectorization
        with open(f"{carpeta_modelo}/text_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_text = json.load(f)

        text_idf = np.load(f"{carpeta_modelo}/text_idf_weights.npy", allow_pickle=True)
        if len(text_idf) == 0:
            text_idf = np.ones(len(vocab_text), dtype=np.float32)

        try:
            instancia.text_vec.set_vocabulary(vocab_text, idf_weights=text_idf.tolist())
        except Exception:
            instancia.text_vec.set_vocabulary(vocab_text[1:], idf_weights=text_idf[1:].tolist())

        # 2) Restaurar vocabulario de StringLookup
        with open(f"{carpeta_modelo}/unit_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_unit = json.load(f)
        try:
            instancia.unit_lookup.set_vocabulary(vocab_unit)
        except Exception:
            instancia.unit_lookup.set_vocabulary(vocab_unit[1:])

        # 3) Construir arquitectura
        instancia.construir()

        # 4) Cargar pesos (usando el nuevo formato .tf)
        instancia.model.load_weights(f"{carpeta_modelo}/pesos_modelo.weights.tf")
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
        x = tf.keras.layers.Dense(self.text_embedding_dim, activation="relu")(x)
        return tf.keras.Model(inp, x, name="text_encoder")

    def construir(self) -> tf.keras.Model:
        # --- Entradas ---
        fact_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_text")
        master_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_text")
        fact_unit = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_unit")
        master_unit = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_unit")
        fact_cost = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_cost")
        master_cost = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_cost")
        fact_peso = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_peso")
        master_peso = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_peso")

        # --- Normalización de Números ---
        self.peso_normalizer = tf.keras.layers.Normalization(axis=-1, name="peso_normalizer")
        fact_peso_norm = self.peso_normalizer(fact_peso)
        master_peso_norm = self.peso_normalizer(master_peso)

        self.cost_normalizer = tf.keras.layers.Normalization(axis=-1, name="cost_normalizer")
        fact_cost_norm = self.cost_normalizer(fact_cost)
        master_cost_norm = self.cost_normalizer(master_cost)

        # --- Similitudes Numéricas ---
        cost_ratio = tf.keras.layers.Lambda(lambda x: x[0] / (x[1] + 1e-6), name="cost_ratio")([fact_cost_norm, master_cost_norm])
        cost_sim = tf.keras.layers.Lambda(lambda x: tf.exp(-tf.abs(x[0] - x[1])), name="cost_sim")([fact_cost_norm, master_cost_norm])
        peso_sim = tf.keras.layers.Lambda(lambda x: tf.exp(-tf.abs(x[0] - x[1])), name="peso_sim")([fact_peso_norm, master_peso_norm])

        # --- Codificación de Texto ---
        text_encoder = self._text_encoder()
        fact_text_vec = text_encoder(fact_text)
        master_text_vec = text_encoder(master_text)

        diff_text = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]), name="diff_text")([fact_text_vec, master_text_vec])
        text_cos = tf.keras.layers.Dot(axes=1, normalize=True, name="text_cos")([fact_text_vec, master_text_vec])

        # --- Unidad ---
        fact_unit_idx = self.unit_lookup(fact_unit)
        master_unit_idx = self.unit_lookup(master_unit)
        unit_match = tf.keras.layers.Lambda(lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32), name="unit_match")([fact_unit_idx, master_unit_idx])

        # --- Concatenación de Características ---
        x = tf.keras.layers.Concatenate()([
            fact_text_vec, master_text_vec, diff_text, text_cos,
            unit_match, fact_cost_norm, master_cost_norm, cost_sim, peso_sim
        ])

        # --- Capas Densas ---
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        
        # Pre-probabilidad (Opinión del modelo)
        logits_match = tf.keras.layers.Dense(1, activation="sigmoid", name="pre_prob")(x)

        # --- COMPUERTAS DE SEGURIDAD (GATES) ---
        # Penaliza si el costo es muy diferente
        cost_gate = tf.keras.layers.Lambda(
            lambda x: tf.exp(-7.0 * tf.square(x - 1.0)), name="cost_gate"
        )(cost_ratio)

        # Penaliza si el peso es muy diferente
        peso_gate = tf.keras.layers.Lambda(
            lambda x: tf.exp(-7.0 * tf.square(x[0] - x[1])), name="peso_gate"
        )([fact_peso_norm, master_peso_norm])

        # Multiplicación final: Si los números fallan, el score baja a 0
        confianza_total = tf.keras.layers.Multiply(name="total_gate")([logits_match, cost_gate, peso_gate])
        y = tf.keras.layers.Activation("linear", name="match_prob")(confianza_total)

        # --- Definición y Compilación ---
        model = tf.keras.Model(
            inputs=[fact_text, master_text, fact_unit, master_unit, fact_cost, master_cost, fact_peso, master_peso],
            outputs=y,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")],
        )

        self.model = model  # <--- CRITICO: Aquí se asigna el modelo para que no sea None
        return model
    
    @staticmethod
    def _to_ds(df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None, batch_size: int = 256, shuffle: bool = False) -> tf.data.Dataset:
        x = {
            "fact_text": df["fact_text"].astype(str).values.reshape(-1, 1),
            "master_text": df["master_text"].astype(str).values.reshape(-1, 1),
            "fact_unit": df["fact_unit"].astype(str).values.reshape(-1, 1),
            "master_unit": df["master_unit"].astype(str).values.reshape(-1, 1),
            "fact_cost": df["fact_cost"].astype(np.float32).values.reshape(-1, 1),
            "master_cost": df["master_cost"].astype(np.float32).values.reshape(-1, 1),
            "fact_peso": df["fact_peso"].astype(np.float32).values.reshape(-1, 1),
            "master_peso": df["master_peso"].astype(np.float32).values.reshape(-1, 1),
        }
        y = df["label"].astype(np.float32).values

        if sample_weight is not None:
            ds = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle:
            ds = ds.shuffle(min(len(df), 10000), seed=42)

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(self, pares: pd.DataFrame, epochs: int = 6, batch_size: int = 256) -> None:
        self.adaptar_vocabularios(pares)

        if self.model is None:
            self.construir()

        # Adaptar normalizador de costos
        all_costs = np.concatenate([pares["fact_cost"].values, pares["master_cost"].values]).reshape(-1, 1)
        self.cost_normalizer.adapt(all_costs)
        
        all_pesos = np.concatenate([pares["fact_peso"].values, pares["master_peso"].values]).reshape(-1, 1)
        self.peso_normalizer.adapt(all_pesos)

        # Dividir en entrenamiento y validación
        train_df, valid_df = train_test_split(pares, test_size=0.2, random_state=42, stratify=pares["label"])

        # Calcular sample weights para entrenamiento
        train_df = train_df.copy()
    
        cost_diff = np.abs(train_df["fact_cost"] - train_df["master_cost"]).values
        peso_diff = np.abs(train_df["fact_peso"] - train_df["master_peso"]).values

        max_cost_diff = cost_diff.max() if cost_diff.max() > 0 else 1.0
        max_peso_diff = peso_diff.max() if peso_diff.max() > 0 else 1.0

        cost_diff_norm = cost_diff / max_cost_diff
        peso_diff_norm = peso_diff / max_peso_diff

        combined_diff = np.maximum(cost_diff_norm, peso_diff_norm)   # énfasis en la mayor discrepancia

        sample_weight = np.ones(len(train_df))
        neg_mask = train_df["label"] == 0
        factor = 10.0
        sample_weight[neg_mask] = 1.0 + factor * combined_diff[neg_mask]

        # Crear datasets
        ds_train = self._to_ds(train_df, sample_weight=sample_weight, batch_size=batch_size, shuffle=True)
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

        # Calcular umbral óptimo en validación
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
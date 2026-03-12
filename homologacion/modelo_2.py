import json
import os
from typing import Optional, List
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


class ModeloMatchCodProducto:
    """
    Clasificador robusto para matching de productos usando deep learning.
    Mejoras respecto a versión anterior:
    - Embeddings de texto y unidades.
    - BatchNormalization y regularización L2.
    - Ingeniería de características extendida.
    - Validación cruzada para umbral.
    - Early stopping basado en F1.
    """

    def __init__(
        self,
        max_tokens: int = 10000,
        text_embedding_dim: int = 128,
        unit_embedding_dim: int = 16,
        text_sequence_length: int = 50,
        dense_units: List[int] = [128, 64],
        dropout_rate: float = 0.4,
        l2_reg: float = 1e-5,
    ):
        self.max_tokens = max_tokens
        self.text_embedding_dim = text_embedding_dim
        self.unit_embedding_dim = unit_embedding_dim
        self.text_sequence_length = text_sequence_length
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        self.text_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=text_sequence_length,
            standardize="lower_and_strip_punctuation",
        )

        self.unit_lookup = tf.keras.layers.StringLookup(
            output_mode="int", mask_token=None
        )

        self.model: Optional[tf.keras.Model] = None
        self.best_threshold: float = 0.5

    def guardar(self, carpeta_modelo: str) -> None:
        """Guarda el modelo completo (pesos, vocabularios, metadatos)."""
        if not os.path.exists(carpeta_modelo):
            os.makedirs(carpeta_modelo)

        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")

        self.model.save_weights(f"{carpeta_modelo}/pesos_modelo.ckpt", save_format='tf')

        vocab_text = self.text_vectorizer.get_vocabulary()
        with open(f"{carpeta_modelo}/text_vocabulary.json", "w", encoding="utf-8") as f:
            json.dump(vocab_text, f, ensure_ascii=False, indent=2)

        vocab_unit = self.unit_lookup.get_vocabulary()
        with open(f"{carpeta_modelo}/unit_vocabulary.json", "w", encoding="utf-8") as f:
            json.dump(vocab_unit, f, ensure_ascii=False, indent=2)

        meta = {
            "max_tokens": self.max_tokens,
            "text_embedding_dim": self.text_embedding_dim,
            "unit_embedding_dim": self.unit_embedding_dim,
            "text_sequence_length": self.text_sequence_length,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "best_threshold": self.best_threshold,
        }
        with open(f"{carpeta_modelo}/meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"Modelo guardado en: {carpeta_modelo}")

    @classmethod
    def cargar(cls, carpeta_modelo: str):
        """Carga un modelo previamente guardado."""
        with open(f"{carpeta_modelo}/meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        instancia = cls(
            max_tokens=meta["max_tokens"],
            text_embedding_dim=meta["text_embedding_dim"],
            unit_embedding_dim=meta["unit_embedding_dim"],
            text_sequence_length=meta["text_sequence_length"],
            dense_units=meta["dense_units"],
            dropout_rate=meta["dropout_rate"],
            l2_reg=meta["l2_reg"],
        )

        with open(f"{carpeta_modelo}/text_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_text = json.load(f)
        instancia.text_vectorizer.set_vocabulary(vocab_text)

        with open(f"{carpeta_modelo}/unit_vocabulary.json", "r", encoding="utf-8") as f:
            vocab_unit = json.load(f)
        instancia.unit_lookup.set_vocabulary(vocab_unit)

        instancia.construir()

        instancia.model.load_weights(f"{carpeta_modelo}/pesos_modelo.ckpt")
        instancia.best_threshold = meta["best_threshold"]

        return instancia

    def adaptar_vocabularios(self, pares: pd.DataFrame) -> None:
        """Adapta las capas de preprocesamiento a los datos."""
        textos = pd.concat(
            [pares["fact_text"], pares["master_text"]], ignore_index=True
        ).astype(str).values

        unidades = pd.concat(
            [pares["fact_unit"], pares["master_unit"]], ignore_index=True
        ).astype(str).values

        self.text_vectorizer.adapt(textos)
        self.unit_lookup.adapt(unidades)

    def _build_text_encoder(self) -> tf.keras.Model:
        """Codificador de texto: Embedding + GlobalAveragePooling."""
        inp = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
        x = self.text_vectorizer(inp)
        x = tf.keras.layers.Embedding(
            input_dim=self.max_tokens + 2,
            output_dim=self.text_embedding_dim,
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="text_embedding",
        )(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(
            self.text_embedding_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.Model(inp, x, name="text_encoder")

    def _build_unit_encoder(self) -> tf.keras.Model:
        """Codificador de unidades: lookup + embedding."""
        inp = tf.keras.Input(shape=(1,), dtype=tf.string, name="unit_input")
        idx = self.unit_lookup(inp)
        emb = tf.keras.layers.Embedding(
            input_dim=self.unit_lookup.vocabulary_size(),
            output_dim=self.unit_embedding_dim,
            embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name="unit_embedding",
        )(idx)
        emb = tf.squeeze(emb, axis=1)
        return tf.keras.Model(inp, emb, name="unit_encoder")

    def construir(self) -> tf.keras.Model:
        """Construye la arquitectura completa del modelo."""
        fact_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_text")
        master_text = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_text")
        fact_unit = tf.keras.Input(shape=(1,), dtype=tf.string, name="fact_unit")
        master_unit = tf.keras.Input(shape=(1,), dtype=tf.string, name="master_unit")
        fact_cost = tf.keras.Input(shape=(1,), dtype=tf.float32, name="fact_cost")
        master_cost = tf.keras.Input(shape=(1,), dtype=tf.float32, name="master_cost")

        text_encoder = self._build_text_encoder()
        unit_encoder = self._build_unit_encoder()

        fact_text_vec = text_encoder(fact_text)
        master_text_vec = text_encoder(master_text)
        fact_unit_vec = unit_encoder(fact_unit)
        master_unit_vec = unit_encoder(master_unit)

        text_diff = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))(
            [fact_text_vec, master_text_vec]
        )
        text_cos = tf.keras.layers.Dot(axes=1, normalize=True)(
            [fact_text_vec, master_text_vec]
        )
        fact_text_len = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.strings.length(x), tf.float32)
        )(fact_text)
        master_text_len = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.strings.length(x), tf.float32)
        )(master_text)
        text_len_diff = tf.abs(fact_text_len - master_text_len)

        unit_cos = tf.keras.layers.Dot(axes=1, normalize=True)(
            [fact_unit_vec, master_unit_vec]
        )
        unit_eq = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32)
        )([fact_unit, master_unit])

        fact_cost_log = tf.keras.layers.Lambda(lambda x: tf.math.log(x + 1e-6))(
            fact_cost
        )
        master_cost_log = tf.keras.layers.Lambda(lambda x: tf.math.log(x + 1e-6))(
            master_cost
        )
        cost_diff = tf.abs(fact_cost_log - master_cost_log)
        cost_ratio = tf.keras.layers.Lambda(
            lambda x: tf.math.abs(x[0] / (x[1] + 1e-6))
        )([fact_cost, master_cost])
        cost_sim = tf.keras.layers.Lambda(lambda x: tf.exp(-x))(cost_diff)

        features = tf.keras.layers.Concatenate()([
            fact_text_vec,
            master_text_vec,
            text_diff,
            text_cos,
            text_len_diff,
            fact_unit_vec,
            master_unit_vec,
            unit_cos,
            unit_eq,
            fact_cost_log,
            master_cost_log,
            cost_diff,
            cost_ratio,
            cost_sim,
        ])

        x = features
        for units in self.dense_units:
            x = tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)

        y = tf.keras.layers.Dense(1, activation="sigmoid", name="match_prob")(x)

        model = tf.keras.Model(
            inputs=[fact_text, master_text, fact_unit, master_unit, fact_cost, master_cost],
            outputs=y,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        self.model = model
        return model

    @staticmethod
    def _to_ds(df: pd.DataFrame, batch_size: int = 256, shuffle: bool = False) -> tf.data.Dataset:
        """Convierte DataFrame a tf.data.Dataset."""
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

    def _f1_score(self, y_true, y_pred):
        """Calcula F1 a partir de arrays numpy."""
        y_pred_bin = (y_pred >= self.best_threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_bin == 1))
        fp = np.sum((y_true == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true == 1) & (y_pred_bin == 0))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return f1

    def fit(
        self,
        pares: pd.DataFrame,
        epochs: int = 20,
        batch_size: int = 256,
        n_splits: int = 3,
        patience: int = 3,
    ) -> None:
        """
        Entrena el modelo con validación cruzada interna para ajustar el umbral.
        """
        self.adaptar_vocabularios(pares)
        if self.model is None:
            self.construir()

        X = pares[["fact_text", "master_text", "fact_unit", "master_unit", "fact_cost", "master_cost"]]
        y = pares["label"].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_preds = []
        fold_true = []

        best_thresholds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold+1}/{n_splits} ---")
            train_df = pares.iloc[train_idx]
            val_df = pares.iloc[val_idx]

            self.construir()

            n_pos = max((train_df["label"] == 1).sum(), 1)
            n_neg = max((train_df["label"] == 0).sum(), 1)
            class_weight = {0: 1.0, 1: n_neg / n_pos}

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
                ),
            ]

            ds_train = self._to_ds(train_df, batch_size=batch_size, shuffle=True)
            ds_val = self._to_ds(val_df, batch_size=batch_size, shuffle=False)

            self.model.fit(
                ds_train,
                validation_data=ds_val,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )

            preds_val = self.model.predict(ds_val, verbose=0).reshape(-1)
            fold_preds.append(preds_val)
            fold_true.append(val_df["label"].values)

            best_f1 = -1.0
            best_th_fold = 0.5
            for th in np.arange(0.3, 0.9, 0.02):
                y_hat = (preds_val >= th).astype(int)
                tp = np.sum((val_df["label"] == 1) & (y_hat == 1))
                fp = np.sum((val_df["label"] == 0) & (y_hat == 1))
                fn = np.sum((val_df["label"] == 1) & (y_hat == 0))
                precision = tp / (tp + fp + 1e-12)
                recall = tp / (tp + fn + 1e-12)
                f1 = 2 * precision * recall / (precision + recall + 1e-12)
                if f1 > best_f1:
                    best_f1 = f1
                    best_th_fold = th
            best_thresholds.append(best_th_fold)
            print(f"Mejor umbral fold {fold+1}: {best_th_fold:.2f} con F1={best_f1:.4f}")

        y_true_all = np.concatenate(fold_true)
        y_pred_all = np.concatenate(fold_preds)

        best_f1_global = -1.0
        best_th_global = 0.5
        for th in np.arange(0.3, 0.9, 0.02):
            f1 = self._f1_score(y_true_all, y_pred_all)
            if f1 > best_f1_global:
                best_f1_global = f1
                best_th_global = th

        self.best_threshold = best_th_global
        print(f"\nUmbral óptimo global: {self.best_threshold:.2f} con F1={best_f1_global:.4f}")

        print("\n--- Entrenamiento final con todos los datos ---")
        self.construir()
        n_pos = max((pares["label"] == 1).sum(), 1)
        n_neg = max((pares["label"] == 0).sum(), 1)
        class_weight = {0: 1.0, 1: n_neg / n_pos}
        ds_all = self._to_ds(pares, batch_size=batch_size, shuffle=True)
        self.model.fit(
            ds_all,
            epochs=epochs,
            class_weight=class_weight,
            verbose=1,
        )

    def predict_pairs(self, pares: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        """Devuelve probabilidades de match para cada par."""
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de predecir.")
        ds = self._to_ds(pares.assign(label=0), batch_size=batch_size, shuffle=False)
        preds = self.model.predict(ds, verbose=0).reshape(-1)
        return preds

    def predict_pairs_binary(self, pares: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Devuelve predicciones binarias usando el umbral almacenado o uno dado."""
        probs = self.predict_pairs(pares)
        th = threshold if threshold is not None else self.best_threshold
        return (probs >= th).astype(int)
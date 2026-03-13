from __future__ import annotations

from dataclasses import asdict
from typing import Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from . import ModelConfig, PreprocessingAssets, ModelPersistence, MatchModelBuilder, ThresholdOptimizer, SampleWeightStrategy, DatasetBuilder

class ModeloMatchProducto:
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
        self.config = ModelConfig(
            max_tokens=max_tokens,
            text_embedding_dim=text_embedding_dim,
            unit_embedding_dim=unit_embedding_dim,
            type_embedding_dim=type_embedding_dim,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
        )

        self.assets = PreprocessingAssets(self.config)
        self.model: Optional[tf.keras.Model] = None
        self.best_threshold: float = 0.78

    @property
    def text_vec(self) -> tf.keras.layers.TextVectorization:
        return self.assets.text_vec

    @property
    def unit_lookup(self) -> tf.keras.layers.StringLookup:
        return self.assets.unit_lookup

    @property
    def type_lookup(self) -> tf.keras.layers.StringLookup:
        return self.assets.type_lookup

    @property
    def cost_normalizer(self) -> tf.keras.layers.Normalization:
        return self.assets.normalizers["cost"]

    @property
    def peso_normalizer(self) -> tf.keras.layers.Normalization:
        return self.assets.normalizers["peso"]

    @property
    def factor_normalizer(self) -> tf.keras.layers.Normalization:
        return self.assets.normalizers["factor"]

    @property
    def content_normalizer(self) -> tf.keras.layers.Normalization:
        return self.assets.normalizers["content"]

    @property
    def total_normalizer(self) -> tf.keras.layers.Normalization:
        return self.assets.normalizers["total"]

    def guardar(self, carpeta_modelo: str) -> None:
        ModelPersistence.save(self, carpeta_modelo)

    @classmethod
    def cargar(cls, carpeta_modelo: str) -> "ModeloMatchProducto":
        config = ModelPersistence.read_config(carpeta_modelo)
        instancia = cls(**asdict(config))
        ModelPersistence.load(instancia, carpeta_modelo)
        return instancia

    def adaptar_vocabularios(self, pares: pd.DataFrame) -> None:
        self.assets.adapt_vocabularies(pares)

    def construir(self) -> tf.keras.Model:
        self.model = MatchModelBuilder(self.config, self.assets).build()
        return self.model

    @staticmethod
    def _to_ds(
        df: pd.DataFrame,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: int = 256,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        return DatasetBuilder.to_dataset(
            df=df,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def fit(self, pares: pd.DataFrame, epochs: int = 12, batch_size: int = 256) -> None:
        groups = (
            pares["RucProveedor"].astype(str).fillna("")
            + "|"
            + pares["fact_cod"].astype(str).fillna("")
            + "|"
            + pares["fact_text"].astype(str).fillna("")
        )

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, valid_idx = next(gss.split(pares, pares["label"], groups=groups))

        train_df = pares.iloc[train_idx].copy().reset_index(drop=True)
        valid_df = pares.iloc[valid_idx].copy().reset_index(drop=True)

        self.adaptar_vocabularios(train_df)

        if self.model is None:
            self.construir()

        self.assets.adapt_normalizers(train_df)

        sample_weight = SampleWeightStrategy.compute(train_df)

        ds_train = self._to_ds(
            train_df,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=True,
        )
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

        best_th, best_score, best_precision, best_recall = ThresholdOptimizer.find_best(
            y_true=y_valid,
            probs=p_valid,
        )

        self.best_threshold = best_th
        print(
            f"Mejor umbral validación: {self.best_threshold:.2f} | "
            f"F1={best_score:.4f} | precision={best_precision:.4f} | recall={best_recall:.4f}"
        )

    def predict_pairs(self, pares: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de predecir.")

        ds = self._to_ds(pares.assign(label=0), batch_size=batch_size, shuffle=False)
        return self.model.predict(ds, verbose=0).reshape(-1)
    

    def split_train_valid(
        self,
        pares: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        groups = (
            pares["RucProveedor"].astype(str).fillna("")
            + "|"
            + pares["fact_cod"].astype(str).fillna("")
            + "|"
            + pares["fact_text"].astype(str).fillna("")
        )

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, valid_idx = next(gss.split(pares, pares["label"], groups=groups))

        train_df = pares.iloc[train_idx].copy().reset_index(drop=True)
        valid_df = pares.iloc[valid_idx].copy().reset_index(drop=True)
        return train_df, valid_df


    def evaluate_pairs(
        self,
        pares: pd.DataFrame,
        batch_size: int = 512,
    ) -> dict:
        if self.model is None:
            raise RuntimeError("Debes entrenar o cargar el modelo antes de evaluar.")

        ds = self._to_ds(pares, batch_size=batch_size, shuffle=False)
        y_true = pares["label"].astype(int).values
        probs = self.model.predict(ds, verbose=0).reshape(-1)

        best_th, best_f1, best_precision, best_recall = ThresholdOptimizer.find_best(
            y_true=y_true,
            probs=probs,
        )

        y_hat_current = (probs >= float(self.best_threshold)).astype(int)

        metrics = {
            "n_samples": int(len(pares)),
            "positive_rate": float(np.mean(y_true)),
            "pr_auc": float(average_precision_score(y_true, probs)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "best_threshold_eval": float(best_th),
            "best_f1_eval": float(best_f1),
            "best_precision_eval": float(best_precision),
            "best_recall_eval": float(best_recall),
            "current_threshold": float(self.best_threshold),
            "current_f1": float(f1_score(y_true, y_hat_current, zero_division=0)),
            "current_precision": float(precision_score(y_true, y_hat_current, zero_division=0)),
            "current_recall": float(recall_score(y_true, y_hat_current, zero_division=0)),
        }
        return metrics


    def fit_incremental_on_split(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        epochs: int = 6,
        batch_size: int = 256,
        recalcular_threshold: bool = True,
    ) -> dict:
        if self.model is None:
            raise RuntimeError(
                "Debes cargar un modelo existente antes de hacer reentrenamiento incremental."
            )

        sample_weight = SampleWeightStrategy.compute(train_df)

        ds_train = self._to_ds(
            train_df,
            sample_weight=sample_weight,
            batch_size=batch_size,
            shuffle=True,
        )
        ds_valid = self._to_ds(
            valid_df,
            batch_size=batch_size,
            shuffle=False,
        )

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

        history = self.model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        if recalcular_threshold:
            y_valid = valid_df["label"].astype(int).values
            p_valid = self.model.predict(ds_valid, verbose=0).reshape(-1)

            best_th, best_score, best_precision, best_recall = ThresholdOptimizer.find_best(
                y_true=y_valid,
                probs=p_valid,
            )

            self.best_threshold = best_th
            print(
                f"Nuevo umbral incremental: {self.best_threshold:.2f} | "
                f"F1={best_score:.4f} | precision={best_precision:.4f} | recall={best_recall:.4f}"
            )

        return {
            "epochs_ran": int(len(history.history.get("loss", []))),
            "history": {
                k: [float(v) for v in vals]
                for k, vals in history.history.items()
            },
        }

    def fit_incremental(
        self,
        pares: pd.DataFrame,
        epochs: int = 6,
        batch_size: int = 256,
        recalcular_threshold: bool = True,
    ) -> None:
        train_df, valid_df = self.split_train_valid(pares)
        self.fit_incremental_on_split(
            train_df=train_df,
            valid_df=valid_df,
            epochs=epochs,
            batch_size=batch_size,
            recalcular_threshold=recalcular_threshold,
        )
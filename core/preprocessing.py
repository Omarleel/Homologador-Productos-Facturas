import numpy as np
import pandas as pd
import tensorflow as tf
from .schema import FeatureSchema
from .config import ModelConfig
from typing import Dict

class PreprocessingAssets:
    def __init__(self, config: ModelConfig):
        self.text_vec = tf.keras.layers.TextVectorization(
            max_tokens=config.max_tokens,
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

        self.normalizers: Dict[str, tf.keras.layers.Normalization] = {
            base: tf.keras.layers.Normalization(axis=-1, name=f"{base}_normalizer")
            for base in FeatureSchema.NUMERIC_BASES
        }

    def adapt_vocabularies(self, pares: pd.DataFrame) -> None:
        textos = pd.concat(
            [
                pares[FeatureSchema.fact("text")],
                pares[FeatureSchema.master("text")],
                pares[FeatureSchema.fact("base_text")],
                pares[FeatureSchema.master("base_text")],
            ],
            ignore_index=True,
        ).astype(str).values

        unidades = pd.concat(
            [pares[FeatureSchema.fact("unit")], pares[FeatureSchema.master("unit")]],
            ignore_index=True,
        ).astype(str).values

        tipos = pd.concat(
            [pares[FeatureSchema.fact("type")], pares[FeatureSchema.master("type")]],
            ignore_index=True,
        ).astype(str).values

        self.text_vec.adapt(textos)
        self.unit_lookup.adapt(unidades)
        self.type_lookup.adapt(tipos)

    def adapt_normalizers(self, train_df: pd.DataFrame) -> None:
        for base in FeatureSchema.NUMERIC_BASES:
            values = np.concatenate(
                [
                    train_df[FeatureSchema.fact(base)].values,
                    train_df[FeatureSchema.master(base)].values,
                ]
            ).reshape(-1, 1)
            self.normalizers[base].adapt(values)


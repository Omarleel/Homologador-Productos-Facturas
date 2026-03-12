
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional
from . import FeatureSchema

class DatasetBuilder:
    @staticmethod
    def to_dataset(
        df: pd.DataFrame,
        sample_weight: Optional[np.ndarray] = None,
        batch_size: int = 256,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        x = {}

        for base in FeatureSchema.TEXT_BASES + FeatureSchema.CATEGORICAL_BASES:
            x[FeatureSchema.fact(base)] = (
                df[FeatureSchema.fact(base)].astype(str).values.reshape(-1, 1)
            )
            x[FeatureSchema.master(base)] = (
                df[FeatureSchema.master(base)].astype(str).values.reshape(-1, 1)
            )

        for base in FeatureSchema.NUMERIC_BASES:
            x[FeatureSchema.fact(base)] = (
                df[FeatureSchema.fact(base)].astype(np.float32).values.reshape(-1, 1)
            )
            x[FeatureSchema.master(base)] = (
                df[FeatureSchema.master(base)].astype(np.float32).values.reshape(-1, 1)
            )

        y = df["label"].astype(np.float32).values

        if sample_weight is not None:
            ds = tf.data.Dataset.from_tensor_slices((x, y, sample_weight))
        else:
            ds = tf.data.Dataset.from_tensor_slices((x, y))

        if shuffle:
            ds = ds.shuffle(min(len(df), 10000), seed=42)

        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

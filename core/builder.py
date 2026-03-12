import tensorflow as tf
from typing import Dict
from . import PreprocessingAssets, FeatureSchema, ModelConfig
class MatchModelBuilder:
    def __init__(self, config: ModelConfig, assets: PreprocessingAssets):
        self.config = config
        self.assets = assets

    def build(self) -> tf.keras.Model:
        inputs = self._build_inputs()

        text_encoder = self._text_encoder()
        unit_encoder = self._categorical_encoder(
            lookup_layer=self.assets.unit_lookup,
            emb_dim=self.config.unit_embedding_dim,
            name="unit",
        )
        type_encoder = self._categorical_encoder(
            lookup_layer=self.assets.type_lookup,
            emb_dim=self.config.type_embedding_dim,
            name="type",
        )

        fact_text_vec = text_encoder(inputs["fact_text"])
        master_text_vec = text_encoder(inputs["master_text"])
        fact_base_vec = text_encoder(inputs["fact_base_text"])
        master_base_vec = text_encoder(inputs["master_base_text"])

        fact_unit_vec = unit_encoder(inputs["fact_unit"])
        master_unit_vec = unit_encoder(inputs["master_unit"])

        fact_type_vec = type_encoder(inputs["fact_type"])
        master_type_vec = type_encoder(inputs["master_type"])

        fact_cost_norm, master_cost_norm = self._normalize_pair("cost", inputs)
        fact_peso_norm, master_peso_norm = self._normalize_pair("peso", inputs)
        fact_factor_norm, master_factor_norm = self._normalize_pair("factor", inputs)
        fact_content_norm, master_content_norm = self._normalize_pair("content", inputs)
        fact_total_norm, master_total_norm = self._normalize_pair("total", inputs)

        diff_text = self._abs_diff(fact_text_vec, master_text_vec, "diff_text")
        diff_base = self._abs_diff(fact_base_vec, master_base_vec, "diff_base")
        diff_unit = self._abs_diff(fact_unit_vec, master_unit_vec, "diff_unit")
        diff_type = self._abs_diff(fact_type_vec, master_type_vec, "diff_type")

        text_cos = self._cosine(fact_text_vec, master_text_vec, "text_cos")
        base_cos = self._cosine(fact_base_vec, master_base_vec, "base_cos")
        unit_cos = self._cosine(fact_unit_vec, master_unit_vec, "unit_cos")
        type_cos = self._cosine(fact_type_vec, master_type_vec, "type_cos")

        fact_unit_idx = self.assets.unit_lookup(inputs["fact_unit"])
        master_unit_idx = self.assets.unit_lookup(inputs["master_unit"])
        unit_match = self._equal_match(fact_unit_idx, master_unit_idx, "unit_match")

        fact_type_idx = self.assets.type_lookup(inputs["fact_type"])
        master_type_idx = self.assets.type_lookup(inputs["master_type"])
        type_match = self._equal_match(fact_type_idx, master_type_idx, "type_match")

        fact_type_known = self._scalar_float_lambda(
            lambda x: tf.cast(
                tf.not_equal(x, tf.constant("NONE", dtype=tf.string)),
                tf.float32,
            ),
            "fact_type_known",
        )(inputs["fact_type"])

        master_type_known = self._scalar_float_lambda(
            lambda x: tf.cast(
                tf.not_equal(x, tf.constant("NONE", dtype=tf.string)),
                tf.float32,
            ),
            "master_type_known",
        )(inputs["master_type"])

        type_available = tf.keras.layers.Multiply(name="type_available")(
            [fact_type_known, master_type_known]
        )

        content_available = self._availability_positive(
            inputs["fact_content"],
            inputs["master_content"],
            "content_available",
        )
        total_available = self._availability_positive(
            inputs["fact_total"],
            inputs["master_total"],
            "total_available",
        )
        peso_available = self._availability_positive(
            inputs["fact_peso"],
            inputs["master_peso"],
            "peso_available",
        )

        cost_diff = self._abs_diff(fact_cost_norm, master_cost_norm, "cost_diff")
        peso_diff = self._abs_diff(fact_peso_norm, master_peso_norm, "peso_diff")
        factor_diff = self._abs_diff(fact_factor_norm, master_factor_norm, "factor_diff")
        content_diff = self._abs_diff(fact_content_norm, master_content_norm, "content_diff")
        total_diff = self._abs_diff(fact_total_norm, master_total_norm, "total_diff")

        cost_sim = self._exp_similarity(fact_cost_norm, master_cost_norm, 1.2, "cost_sim")
        peso_sim = self._exp_similarity(fact_peso_norm, master_peso_norm, 1.8, "peso_sim")
        factor_sim = self._exp_similarity(fact_factor_norm, master_factor_norm, 2.0, "factor_sim")
        content_sim = self._exp_similarity(fact_content_norm, master_content_norm, 2.0, "content_sim")
        total_sim = self._exp_similarity(fact_total_norm, master_total_norm, 2.2, "total_sim")

        factor_close = self._scalar_float_lambda(
            lambda x: tf.cast(tf.abs(x[0] - x[1]) <= 0.10, tf.float32),
            "factor_close",
        )([inputs["fact_factor"], inputs["master_factor"]])

        content_close = self._scalar_float_lambda(
            lambda x: tf.cast(tf.abs(x[0] - x[1]) <= 0.12, tf.float32),
            "content_close",
        )([inputs["fact_content"], inputs["master_content"]])

        total_close = self._scalar_float_lambda(
            lambda x: tf.cast(tf.abs(x[0] - x[1]) <= 0.12, tf.float32),
            "total_close",
        )([inputs["fact_total"], inputs["master_total"]])

        peso_rel_diff = tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]) / tf.maximum(
                tf.maximum(tf.abs(x[0]), tf.abs(x[1])),
                1e-6,
            ),
            name="peso_rel_diff",
        )([inputs["fact_peso"], inputs["master_peso"]])

        peso_close = self._scalar_float_lambda(
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

        lexical_repr = self._dense_block(
            lexical_features,
            units=(192, 96),
            names=("lexical_dense_1", "lexical_dense_2"),
        )

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

        presentation_repr = self._dense_block(
            presentation_features,
            units=(160, 64),
            names=("presentation_dense_1", "presentation_dense_2"),
        )

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
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="fusion_dense_1",
        )(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate + 0.05)(x)
        x = tf.keras.layers.Dense(
            96,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name="fusion_dense_2",
        )(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)

        _ = tf.keras.layers.Dense(1, activation="sigmoid", name="pre_prob")(x)

        type_gate_base = self._scalar_float_lambda(
            lambda z: 0.08 + 0.92 * z,
            "type_gate_base",
        )(type_match)

        type_gate = self._scalar_float_lambda(
            lambda z: (z[0] * z[1]) + ((1.0 - z[0]) * 1.0),
            "type_gate",
        )([type_available, type_gate_base])

        factor_gate = self._scalar_float_lambda(
            lambda z: 0.02 + 0.98 * (0.80 * z[0] + 0.20 * z[1]),
            "factor_gate",
        )([factor_sim, factor_close])

        content_gate_base = self._scalar_float_lambda(
            lambda z: 0.05 + 0.95 * (0.65 * z[0] + 0.25 * z[1] + 0.10 * z[2]),
            "content_gate_base",
        )([content_sim, content_close, type_match])

        content_gate = self._scalar_float_lambda(
            lambda z: (z[0] * z[1]) + ((1.0 - z[0]) * 1.0),
            "content_gate",
        )([content_available, content_gate_base])

        total_gate_base = self._scalar_float_lambda(
            lambda z: 0.02 + 0.98 * (0.70 * z[0] + 0.20 * z[1] + 0.10 * z[2]),
            "total_gate_base",
        )([total_sim, total_close, type_match])

        total_gate = self._scalar_float_lambda(
            lambda z: (z[0] * z[1]) + ((1.0 - z[0]) * 1.0),
            "total_gate",
        )([total_available, total_gate_base])

        peso_gate_base = self._scalar_float_lambda(
            lambda z: 0.10 + 0.90 * (0.70 * z[0] + 0.30 * z[1]),
            "peso_gate_base",
        )([peso_sim, peso_close])

        peso_gate = self._scalar_float_lambda(
            lambda z: (z[0] * z[1]) + ((1.0 - z[0]) * 1.0),
            "peso_gate",
        )([peso_available, peso_gate_base])

        gate_mean = self._scalar_float_lambda(
            lambda z: (
                0.20 * z[0]
                + 0.28 * z[1]
                + 0.18 * z[2]
                + 0.22 * z[3]
                + 0.12 * z[4]
            ),
            "presentation_gate_mean",
        )([type_gate, factor_gate, content_gate, total_gate, peso_gate])

        gate_min = self._scalar_float_lambda(
            lambda z: tf.reduce_min(tf.concat(z, axis=1), axis=1, keepdims=True),
            "presentation_gate_min",
        )([factor_gate, content_gate, total_gate, type_gate])

        presentation_gate = self._scalar_float_lambda(
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
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
        )(fusion2)
        fusion2 = tf.keras.layers.Dropout(self.config.dropout_rate)(fusion2)

        y = tf.keras.layers.Dense(1, activation="sigmoid", name="match_prob")(fusion2)

        model = tf.keras.Model(inputs=list(inputs.values()), outputs=y)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.BinaryAccuracy(name="acc"),
            ],
        )
        return model

    def _build_inputs(self) -> Dict[str, tf.keras.Input]:
        inputs: Dict[str, tf.keras.Input] = {}

        for base in FeatureSchema.TEXT_BASES + FeatureSchema.CATEGORICAL_BASES:
            inputs[FeatureSchema.fact(base)] = tf.keras.Input(
                shape=(1,),
                dtype=tf.string,
                name=FeatureSchema.fact(base),
            )
            inputs[FeatureSchema.master(base)] = tf.keras.Input(
                shape=(1,),
                dtype=tf.string,
                name=FeatureSchema.master(base),
            )

        for base in FeatureSchema.NUMERIC_BASES:
            inputs[FeatureSchema.fact(base)] = tf.keras.Input(
                shape=(1,),
                dtype=tf.float32,
                name=FeatureSchema.fact(base),
            )
            inputs[FeatureSchema.master(base)] = tf.keras.Input(
                shape=(1,),
                dtype=tf.float32,
                name=FeatureSchema.master(base),
            )

        return inputs

    def _normalize_pair(
        self,
        base: str,
        inputs: Dict[str, tf.keras.Input],
    ) -> tuple[tf.Tensor, tf.Tensor]:
        normalizer = self.assets.normalizers[base]
        return (
            normalizer(inputs[FeatureSchema.fact(base)]),
            normalizer(inputs[FeatureSchema.master(base)]),
        )

    def _text_encoder(self) -> tf.keras.Model:
        inp = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
        x = self.assets.text_vec(inp)
        x = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        x = tf.keras.layers.Dense(
            self.config.text_embedding_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
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
            embeddings_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_embedding",
        )(idx)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            emb_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=f"{name}_dense",
        )(x)
        return tf.keras.Model(inp, x, name=f"{name}_encoder")

    def _dense_block(
        self,
        inputs: tf.Tensor,
        units: tuple[int, int],
        names: tuple[str, str],
    ) -> tf.Tensor:
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.Dense(
            units[0],
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=names[0],
        )(x)
        x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        x = tf.keras.layers.Dense(
            units[1],
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg),
            name=names[1],
        )(x)
        return x

    @staticmethod
    def _abs_diff(a: tf.Tensor, b: tf.Tensor, name: str) -> tf.Tensor:
        return tf.keras.layers.Lambda(
            lambda x: tf.abs(x[0] - x[1]),
            name=name,
        )([a, b])

    @staticmethod
    def _cosine(a: tf.Tensor, b: tf.Tensor, name: str) -> tf.Tensor:
        return tf.keras.layers.Dot(
            axes=1,
            normalize=True,
            name=name,
        )([a, b])

    @staticmethod
    def _equal_match(a: tf.Tensor, b: tf.Tensor, name: str) -> tf.Tensor:
        return tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.equal(x[0], x[1]), tf.float32),
            name=name,
        )([a, b])

    @staticmethod
    def _exp_similarity(a: tf.Tensor, b: tf.Tensor, factor: float, name: str) -> tf.Tensor:
        return tf.keras.layers.Lambda(
            lambda x: tf.exp(-factor * tf.abs(x[0] - x[1])),
            name=name,
        )([a, b])

    @staticmethod
    def _availability_positive(a: tf.Tensor, b: tf.Tensor, name: str) -> tf.Tensor:
        return MatchModelBuilder._scalar_float_lambda(
            lambda x: tf.cast(
                tf.logical_and(x[0] > 0.0, x[1] > 0.0),
                tf.float32,
            ),
            name,
        )([a, b])

    @staticmethod
    def _scalar_float_lambda(fn, name: str) -> tf.keras.layers.Layer:
        return tf.keras.layers.Lambda(
            fn,
            output_shape=(1,),
            dtype=tf.float32,
            name=name,
        )


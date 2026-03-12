from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    max_tokens: int = 12000
    text_embedding_dim: int = 48
    unit_embedding_dim: int = 8
    type_embedding_dim: int = 4
    dropout_rate: float = 0.20
    l2_reg: float = 1e-4
    learning_rate: float = 8e-4
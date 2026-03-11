import numpy as np
import tensorflow as tf

SEED = 42


def init_seeds() -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
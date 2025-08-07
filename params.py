from flax import nnx
import jax.numpy as jnp
from jax import random
from schedule import cosine_beta_schedule


DTYPE = jnp.float32

B = 8
EPOCHS = 60

T = 200

T_dim = 128
T_hidden = 1024
T_out = 128

TEXT_EMBEDDING_DIM = 384

H = 64
W = 64

CHANNEL_SAMPLING_FACTOR = 2
BASE_DIM = 32

RNGS = nnx.Rngs(params=random.key(32))


SCHEDULE = cosine_beta_schedule(T)
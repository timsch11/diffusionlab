from flax import nnx
import jax.numpy as jnp
from jax import random
from schedule import cosine_beta_schedule


"""Global parameters for current training setup and inference"""

# datatype for model and data
DTYPE = jnp.float32

# Batch size
B = 24

# Train epochs
EPOCHS = 500

# Denoising timesteps
T = 250

# Config for timestamp embedding model
T_dim = 128
T_out = 128

# Dim for prompt embedding
TEXT_EMBEDDING_DIM = 384

# desired height and width for training
H = 64
W = 64

# Factor for up/downsampling
CHANNEL_SAMPLING_FACTOR = 2

# Base channel dimension
BASE_DIM = 24

# Random seeds
RNGS = nnx.Rngs(params=random.key(32))
RANDOMKEY = random.PRNGKey(42)

# Schedule
SCHEDULE = cosine_beta_schedule(T)

# Controlls share of the dataset to use
MAX_INDEX = 1
from flax import nnx
import jax.numpy as jnp
from jax import random
from schedule import cosine_beta_schedule


"""Global place for parameters for current training setup and inference"""


# datatype for model and data
DTYPE = jnp.float32

# Batch size
B = 6

# Train epochs
EPOCHS = 750

# Denoising timesteps
T = 200

# Config for timestamp embedding model
T_dim = 256
T_out = 512

# Dim for prompt embedding
TEXT_EMBEDDING_DIM = 512

# desired height and width for training
H = 64
W = 64

# Factor for up/downsampling
CHANNEL_SAMPLING_FACTOR = 2

# Base channel dimension
BASE_DIM = 20

# Random seeds
RNGS = nnx.Rngs(params=random.key(32))
RANDOMKEY = random.PRNGKey(42)

# Schedule
SCHEDULE = cosine_beta_schedule(T)

# Controlls share of the dataset to use
MAX_INDEX = 150  # -1 for whole dataset

# File to store mean and std of dataset to
DATASET_MEASURE_FILE = "dataset_stats.npz"
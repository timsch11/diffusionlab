#import os
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


from diffusion.model import DiffusionNet
"""from schedule import cosine_beta_schedule
from dataloader import Dataloader

from util import save_model, load_model"""

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_hidden, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM

import jax.numpy as jnp

t = jnp.stack([1, 2, -2, 1])
print(t.shape)
t_array = jnp.full((4,), t, dtype=DTYPE)

print(t_array.shape)
#from jax import random
#from flax import nnx
#from tqdm import tqdm
#import jax
#import optax

B = 160

model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_hidden=T_hidden, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)

x = jnp.full(shape=(B, H, W, 3), fill_value=0.323)
c = jnp.full(shape=(B, 1, 384), fill_value=0.3443)
t = jnp.stack([i for i in range(B)])


result = model(x, t, c)

print(result)
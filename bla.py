import jax
import time
import random
import jax.numpy as jnp
from tqdm import tqdm
from flax import nnx


dim = 100

@nnx.jit
def get_timestep_embedding_jitted(timesteps: int, dtype: jnp.dtype = jnp.bfloat16):
    """
    Embedds timestemp using sinusoidal encoding
    
    Parameters:
        timesteps (int): timestamp
        dim (int): embedding dimension 
    """

    # wrap into array if neccessary
    if not isinstance(timesteps, jax.Array):
        timesteps = jnp.array([timesteps], dtype=dtype)

    # calc exponents
    half_dim = dim // 2
    exponents = jnp.arange(half_dim, dtype=dtype) / half_dim

    # calc frequencies and angles
    freqs = 10000 ** (-exponents)  # [half_dim]
    angles = timesteps[:, None] * freqs[None, :]  # [B, half_dim]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], dtype=dtype, axis=-1)


def get_timestep_embedding(timesteps: int, dim: int, dtype: jnp.dtype = jnp.bfloat16):
    """
    Embedds timestemp using sinusoidal encoding
    
    Parameters:
        timesteps (int): timestamp
        dim (int): embedding dimension 
    """

    # wrap into array if neccessary
    if not isinstance(timesteps, jax.Array):
        timesteps = jnp.array([timesteps], dtype=dtype)

    # calc exponents
    half_dim = dim // 2
    exponents = jnp.arange(half_dim, dtype=dtype) / half_dim

    # calc frequencies and angles
    freqs = 10000 ** (-exponents)  # [half_dim]
    angles = timesteps[:, None] * freqs[None, :]  # [B, half_dim]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], dtype=dtype, axis=-1)


r = get_timestep_embedding_jitted(jnp.array([1]))


t2 = time.time()
for i in range(2, 1000):
    r += get_timestep_embedding_jitted(jnp.array([i]))

print(f"jitted time: {time.time() - t2}")
print(r)

get_timestep_embedding_2 = nnx.jit(get_timestep_embedding, static_argnames=("dim",))
get_timestep_embedding_2(jnp.array([[1, 2]]), dim)
r = get_timestep_embedding_2(jnp.array([1]), dim)

t = time.time()
for i in range(2, 1000):
    r += get_timestep_embedding_2(jnp.array([i]), dim)

print(f"Non jitted time: {time.time() - t}")
print(r)



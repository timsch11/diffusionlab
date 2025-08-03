from flax import nnx
from jax import Array
import jax.numpy as jnp


class TimestampNet(nnx.Module):
    """
    Projects timestamp embedding into the model
    2 layer perceptron with silu
    """

    def __init__(self, inital_embedding_size: int, hidden_size: int, target_size: int, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.inital_dim = inital_embedding_size
        self.linear1 = nnx.Linear(in_features=inital_embedding_size, out_features=hidden_size, dtype=dtype, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=hidden_size, out_features=target_size, dtype=dtype, rngs=rngs)
        self.get_timestep_embedding_jitted = nnx.jit(get_timestep_embedding, static_argnames=("dim", "dtype"))
        self.norm = nnx.LayerNorm(num_features=target_size, dtype=dtype, rngs=rngs)

    @nnx.jit
    def __call__(self, t: int) -> Array:
        x = self.get_timestep_embedding_jitted(t, dim=self.inital_dim, dtype=self.dtype)
        return self.norm(self.linear2(nnx.silu(self.linear1(x))))
    

def get_timestep_embedding(timesteps: int, dim: int, dtype: jnp.dtype = jnp.bfloat16):
    """
    Embedds timestemp using sinusoidal encoding
    
    Parameters:
        timesteps (int): timestamp
        dim (int): embedding dimension 
    """

    # wrap into array if neccessary
    if not isinstance(timesteps, Array):
        timesteps = jnp.array([timesteps], dtype=dtype)

    # calc exponents
    half_dim = dim // 2
    exponents = jnp.arange(half_dim, dtype=dtype) / half_dim

    # calc frequencies and angles
    freqs = 10000 ** (-exponents)  # [half_dim]
    angles = timesteps[:, None] * freqs[None, :]  # [B, half_dim]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], dtype=dtype, axis=-1)

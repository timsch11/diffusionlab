from flax import nnx
from jax import Array
import jax.numpy as jnp


class TimestampNet(nnx.Module):
    """
    Projects timestamp embedding into the model
    2 layer perceptron with silu
    """

    def __init__(self, inital_embedding_size: int, hidden_size: int, target_size: int, rngs: nnx.Rngs):
        super().__init__()
        self.inital_dim = inital_embedding_size
        self.linear1 = nnx.Linear(in_features=inital_embedding_size, out_features=hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=hidden_size, out_features=target_size, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=target_size, rngs=rngs)

    def __call__(self, t: int) -> Array:
        x = get_timestep_embedding(t, dim=self.inital_dim)
        return self.norm(self.linear2(nnx.silu(self.linear1(x))))
    

def get_timestep_embedding(timesteps: int, dim: int):
    """
    Embedds timestemp using sinusoidal encoding
    
    Parameters:
        timesteps (int): timestamp
        dim (int): embedding dimension 
    """

    # wrap into array if neccessary
    if not isinstance(timesteps, Array):
        timesteps = jnp.array([timesteps])

    # calc exponents
    half_dim = dim // 2
    exponents = jnp.arange(half_dim) / half_dim

    # calc frequencies and angles
    freqs = 10000 ** (-exponents)  # [half_dim]
    angles = timesteps[:, None] * freqs[None, :]  # [B, half_dim]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

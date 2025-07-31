from flax import nnx
import jax.numpy as jnp
from jax import Array


class DiffusionNet(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(din, dout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return self.linear1(x)
  

class TimestampNet(nnx.Module):
    """
    Projects timestamp embedding into the model
    2 layer perceptron with silu
    """

    def __init__(self, inital_embedding_size: int, hidden_size: int, target_size: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(in_features=inital_embedding_size, out_features=hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=hidden_size, out_features=target_size, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        return nnx.LayerNorm(self.linear2(nnx.silu(self.linear1(x))))


class DownsampleBlock(nnx.Module):
    def __init__(self, inp_shape: tuple[int, int], in_features: int, out_features: int, timestamp_embedding_size: int, rngs: nnx.Rngs, self_attention: bool = False):
        super().__init__()
        self.timestamp_embedding_projection = nnx.Linear(in_features=timestamp_embedding_size, out_features=inp_shape[0]*inp_shape[1], rngs=rngs)
        self.is_self_attention = self_attention
        self.conv1 = nnx.Conv(in_features=in_features, out_features=out_features, padding=1,kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_features, out_features=out_features, padding=1, strides=2, kernel_size=(3, 3), rngs=rngs)
        self.rngs = rngs

    def __call__(self, x, t) -> Array:
        # Residual Block
        s1 = self.conv1(nnx.silu(nnx.LayerNorm(x, rngs=self.rngs)))                         # conv 1
        s2 = s1 + self.timestamp_embedding_projection(t)[:, None, None, :]  # project timestamp embedding
        s3 = self.conv1(nnx.silu(nnx.LayerNorm(s2, rngs=self.rngs)))                        # conv 2
        r = s3 + x

        # Self attention (optionally)
        if self.is_self_attention:
            pass

        return r



def get_timestep_embedding(timesteps: int, dim: int):
    """
    Embedds timestemp using sinusoidal encoding
    
    Parameters:
        timesteps (int): timestamp
        dim (int): embedding dimension 
    """

    # calc exponents
    half_dim = dim // 2
    exponents = jnp.arange(half_dim) / half_dim

    # calc frequencies and angles
    freqs = 10000 ** (-exponents)  # [half_dim]
    angles = timesteps[:, None] * freqs[None, :]  # [B, half_dim]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


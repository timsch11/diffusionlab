from flax import nnx
import jax.numpy as jnp
from jax import Array


class ResNet(nnx.Module):
    def __init__(self, channels: int, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        super().__init__()

        # conv layers
        self.conv1 = nnx.Conv(in_features=channels, out_features=channels, padding=1, kernel_size=(3, 3), dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=channels, out_features=channels, padding=1, kernel_size=(3, 3), dtype=dtype, rngs=rngs)
        self.rngs = rngs

        # norm layers
        norm_groups = 3 if channels < 24 else 24

        self.norm1 = nnx.GroupNorm(num_groups=norm_groups, num_features=channels, dtype=dtype, rngs=rngs)
        self.norm2 = nnx.GroupNorm(num_groups=norm_groups, num_features=channels, dtype=dtype, rngs=rngs)

    @nnx.jit
    def __call__(self, x, t_embedd) -> Array:
        ### Residual Block
        ## Conv
        s1 = self.conv1(nnx.silu(self.norm1(x)))                                    # conv 1
        s2 = s1 + t_embedd[:, None, None, :]                                        # add timestamp embedding
        s3 = self.conv2(nnx.silu(self.norm2(s2)))                                   # conv 2     

        ## Add residual connection
        r = s3 + x                                        # residual connection   

        return r
    
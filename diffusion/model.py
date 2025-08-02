from flax import nnx
import jax.numpy as jnp
from jax import Array, image


from diffusion.blocks.encoder import DownsampleBlock
from diffusion.blocks.decoder import UpsampleBlock
from diffusion.blocks.timestamp_encoding import TimestampNet


class DiffusionNet(nnx.Module):
    def __init__(self, height: int, width: int, channels: int, channel_sampling_factor: int, t_in: int, t_hidden: int, t_out: int, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16):
        """
        Initalizes a Diffusion U-Net of the given configuration.
        """
        
        super().__init__()
        
        ### config
        ## timestamp encoding
        self.timestamp_net = TimestampNet(t_in, t_hidden, t_out, dtype=dtype, rngs=rngs)

        ## Encoder (downsampling)
        # shapes are examples for height=128, width=128, channels=3 

        # [B, 128, 128, 3] -> [B, 64, 64, 6]
        self.d1 = DownsampleBlock(height, width, in_channels=channels, out_channels=channels*channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs) 

        # adjust height and width
        height = -(height // -2)    # ceildiv
        width = -(width // -2)      # ceildiv
        channels *= channel_sampling_factor

        # [B, 64, 64, 6] -> [B, 32, 32, 12]
        self.d2 = DownsampleBlock(height, width, in_channels=channels, out_channels=channels*channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=3) 

        # adjust height and width
        height = -(height // -2)    # ceildiv
        width = -(width // -2)      # ceildiv
        channels *= channel_sampling_factor

        # [B, 32, 32, 12] -> [B, 16, 16, 24]
        self.d3 = DownsampleBlock(height, width, in_channels=channels, out_channels=channels*channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=3) 

        # adjust height and width
        height = -(height // -2)    # ceildiv
        width = -(width // -2)      # ceildiv
        channels *= channel_sampling_factor

        # [B, 16, 16, 24] -> [B, 8, 8, 48]
        self.d4 = DownsampleBlock(height, width, in_channels=channels, out_channels=channels*channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=3) 

        # adjust height and width
        height = -(height // -2)    # ceildiv
        width = -(width // -2)      # ceildiv
        channels *= channel_sampling_factor


        ## Decoder (downsampling)
        # shapes starting from height=8, width=8, channels=48

        # [B, 8, 8, 48] -> [B, 16, 16, 24]
        self.u1 = UpsampleBlock(height, width, in_channels=channels, out_channels=channels // channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=3) 

        # adjust height and width
        height *= 2
        width *= 2
        channels //= channel_sampling_factor

        # [B, 16, 16, 24] -> [B, 32, 32, 12]
        self.u2 = UpsampleBlock(height, width, in_channels=channels, out_channels=channels // channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=3) 

        # adjust height and width
        height *= 2
        width *= 2
        channels //= channel_sampling_factor

        # [B, 32, 32, 12] -> [B, 64, 64, 6]
        self.u3 = UpsampleBlock(height, width, in_channels=channels, out_channels=channels // channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=3) 

        # adjust height and width
        height *= 2
        width *= 2
        channels //= channel_sampling_factor

        # [B, 64, 64, 6] -> [B, 128, 128, 3]
        self.u4 = UpsampleBlock(height, width, in_channels=channels, out_channels=channels // channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs) 

    def __call__(self, x: Array, t: int, c: Array = None) -> Array:
        # embedd timestamp
        timestamp_embedding = self.timestamp_net(t)

        # downsampling 1
        x_d_1 = self.d1(x, timestamp_embedding)

        # downsampling 1
        x_d_2 = self.d2(x_d_1, timestamp_embedding)

        # downsampling 1
        x_d_3 = self.d3(x_d_2, timestamp_embedding)

        # downsampling 1
        x_d_4 = self.d4(x_d_3, timestamp_embedding)

        # upsampling 1
        x_u_1 = self.u1(x_d_4, x_skip=x_d_3, t=timestamp_embedding)

        # upsampling 2
        x_u_2 = self.u2(x_u_1, x_skip=x_d_2, t=timestamp_embedding)

        # upsampling 3
        x_u_3 = self.u3(x_u_2, x_skip=x_d_1, t=timestamp_embedding)

        # upsampling 4
        x_u_4 = self.u4(x_u_3, x_skip=x, t=timestamp_embedding)

        return x_u_4


        
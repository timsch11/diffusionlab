from flax import nnx
import jax.numpy as jnp
from jax import Array, image


from diffusion.blocks.encoder import DownsampleBlock
from diffusion.blocks.decoder import UpsampleBlock, UpsampleBlockAtt
from diffusion.blocks.bottleneck import BottleneckBlock
from diffusion.blocks.timestamp_encoding import TimestampNet


class DiffusionNet(nnx.Module):
    def __init__(self, height: int, width: int, channels: int, base_dim: int, channel_sampling_factor: int, t_in: int, t_hidden: int, t_out: int, text_embedding_dim: int, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16):
        """
        Initalizes a Diffusion U-Net of the given configuration.

        """
        
        super().__init__()

        # base dim conv
        self.conv1 = nnx.Conv(in_features=channels, out_features=base_dim, kernel_size=1, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=base_dim, out_features=channels, kernel_size=1, dtype=dtype, rngs=rngs)
        
        ### config
        ## timestamp encoding
        self.timestamp_net = TimestampNet(t_in, t_hidden, t_out, dtype=dtype, rngs=rngs)

        ## Encoder (downsampling)
        # shapes are examples for height=128, width=128, channels=3 

        in_channels = channels + 0

        channels = base_dim * channel_sampling_factor

        # [B, 128, 128, 3] -> [B, 64, 64, 60]
        self.d1 = DownsampleBlock(height, width, in_channels=base_dim, out_channels=channels, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs) 

        # adjust height and width
        height = -(height // -2)    # ceildiv
        width = -(width // -2)      # ceildiv

        # [B, 64, 64, 60] -> [B, 32, 32, 120]
        self.d2 = DownsampleBlock(height, width, in_channels=channels, out_channels=channels*channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs) 

        # adjust height and width
        height = -(height // -2)    # ceildiv
        width = -(width // -2)      # ceildiv
        channels *= channel_sampling_factor

        # [B, 32, 32, 120] -> [B, 16, 16, 240]
        self.d3 = DownsampleBlock(height, width, in_channels=channels, out_channels=channels*channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs) 

        # adjust height and width
        height = -(height // -2)    # ceildiv
        width = -(width // -2)      # ceildiv
        channels *= channel_sampling_factor

        # [B, 16, 16, 240] -> [B, 8, 8, 480]
        self.d4 = DownsampleBlock(height, width, in_channels=channels, out_channels=channels*channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=4) 

        # adjust height and width
        height = -(height // -2)    # ceildiv
        width = -(width // -2)      # ceildiv
        channels *= channel_sampling_factor

        # Bottleneck
        # 
        self.b1 = BottleneckBlock(height, width, channels=channels, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=4, cross_attention=True, cross_attention_heads=4, text_embedding_dim=text_embedding_dim) 

        ## Decoder (downsampling)
        # shapes starting from height=8, width=8, channels=48

        # [B, 8, 8, 480] -> [B, 16, 16, 240]
        self.u1 = UpsampleBlock(height, width, in_channels=channels, out_channels=channels // channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs, self_attention=True, self_attention_heads=4) 

        # adjust height and width
        height *= 2
        width *= 2
        channels //= channel_sampling_factor

        # [B, 16, 16, 240] -> [B, 32, 32, 120]
        self.u2 = UpsampleBlock(height, width, in_channels=channels, out_channels=channels // channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs) 

        # adjust height and width
        height *= 2
        width *= 2
        channels //= channel_sampling_factor

        # [B, 32, 32, 120] -> [B, 64, 64, 60]
        self.u3 = UpsampleBlock(height, width, in_channels=channels, out_channels=channels // channel_sampling_factor, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs) 

        # adjust height and width
        height *= 2
        width *= 2
        channels //= channel_sampling_factor

        # [B, 64, 64, 60] -> [B, 128, 128, 3]
        self.u4 = UpsampleBlock(height, width, in_channels=channels, out_channels=base_dim, timestamp_embedding_size=t_out, dtype=dtype, rngs=rngs) 

    @nnx.jit
    def __call__(self, x: Array, t: Array, c: Array = None) -> Array:
        # embedd timestamp
        timestamp_embedding = self.timestamp_net(t)

        # channels -> base dim
        x = self.conv1(x)

        # downsampling 1
        x_d_1, x_skip_1 = self.d1(x, timestamp_embedding)

        # downsampling 1
        x_d_2, x_skip_2 = self.d2(x_d_1, timestamp_embedding)

        # downsampling 1
        x_d_3, x_skip_3 = self.d3(x_d_2, timestamp_embedding, c)

        # downsampling 1
        x_d_4, x_skip_4 = self.d4(x_d_3, timestamp_embedding, c)

        # bottleneck
        x_b = self.b1(x_d_4, timestamp_embedding, c)

        # upsampling 1
        x_u_1 = self.u1(x_b, x_skip=x_skip_4, t=timestamp_embedding, c=c)

        # upsampling 2
        x_u_2 = self.u2(x_u_1, x_skip=x_skip_3, t=timestamp_embedding)

        # upsampling 3
        x_u_3 = self.u3(x_u_2, x_skip=x_skip_2, t=timestamp_embedding)

        # upsampling 4
        x_u_4 = self.u4(x_u_3, x_skip=x_skip_1, t=timestamp_embedding)

        # base dim -> channels
        r = self.conv2(x_u_4)

        return r

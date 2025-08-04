from flax import nnx
import jax.numpy as jnp
from jax import Array, image


class UpsampleBlock(nnx.Module):
    def __init__(self, height: int, width: int, in_channels: int, out_channels: int, timestamp_embedding_size: int, rngs: nnx.Rngs, self_attention: bool = False, self_attention_heads: int = 2, cross_attention: bool = False, cross_attention_heads: int = 2, text_embedding_dim: int = None, dtype: jnp.dtype = jnp.bfloat16):
        super().__init__()

        # calculate downsampling factor
        if in_channels % out_channels != 0:
            raise ValueError(f"Invalid values for in_channels and out_channels, must evenly divide, values: in_channels: {in_channels}, out_channels: {out_channels}")
        
        self.upsampling_factor = in_channels // out_channels

        # projection of timestamp embedding into channels
        self.timestamp_embedding_projection = nnx.Linear(in_features=timestamp_embedding_size, out_features=out_channels, dtype=dtype, rngs=rngs)

        # conv layers
        self.conv1 = nnx.Conv(in_features=out_channels*(self.upsampling_factor+1), out_features=out_channels, kernel_size=1, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_channels, out_features=out_channels, padding=1, kernel_size=(3, 3), dtype=dtype, rngs=rngs)
        self.rngs = rngs

        # norm layers
        self.norm1 = nnx.LayerNorm(num_features=out_channels, feature_axes=-1, dtype=dtype, rngs=rngs)

        # whether to apply self/cross attention
        self.is_self_attention = self_attention
        self.is_cross_attention = cross_attention

        # initalize attention layers if neccessary
        # self attention
        if self_attention:
            # check for valid amount of attention heads
            if self_attention_heads < 1:
                raise ValueError("Invalid attention head parameter for self attention")
            
            self.self_attention = nnx.MultiHeadAttention(in_features=out_channels, num_heads=self_attention_heads, qkv_features=out_channels, decode=False, dtype=dtype, rngs=rngs)
            self.self_attention_pos_embedding = nnx.Param(value=jnp.full(shape=(1, height * width * 4, out_channels), fill_value=0.0, dtype=dtype))

        # cross attention
        if cross_attention:
            # check for valid amount of attention heads
            if cross_attention_heads < 1:
                raise ValueError("Invalid attention head parameter for cross attention")
            
            if text_embedding_dim is None:
                raise ValueError("Set text embedding dimension if you want to use cross attention")
            
            self.cross_attention = nnx.MultiHeadAttention(in_features=out_channels, in_kv_features=text_embedding_dim, num_heads=cross_attention_heads, decode=False, dtype=dtype, rngs=rngs)
            self.cross_attention_pos_embedding = nnx.Param(value=jnp.full(shape=(1, height * width * 4, out_channels), fill_value=0.0, dtype=dtype))

    @nnx.jit
    def __call__(self, x, x_skip, t, c=None) -> Array:
        ### Upsampling (nearest neighbor)
        x = image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method="nearest")

        ### Concat skip connection
        x = jnp.concatenate((x, x_skip), axis=-1)

        ### Residual Block
        ## Conv
        x = self.conv1(x)                                                           # conv 1
        s2 = x + self.timestamp_embedding_projection(t)[:, None, None, :]           # project timestamp embedding
        s3 = self.conv2(nnx.silu(self.norm1(s2)))                                   # conv 2             

        ## Attention (optionally)
        # Combines self and cross attention if possible to enhance efficience
        if self.is_self_attention and self.is_cross_attention:
            reshaped_s3 = s3.reshape(-1, s3.shape[1]*s3.shape[2], s3.shape[3])                              # reshape to [B, H*W, C]
            reshaped_self_embedding = reshaped_s3 + self.self_attention_pos_embedding                       # embedd positional encoding for self attention
            self_att = self.self_attention(reshaped_self_embedding, reshaped_self_embedding)
            reshaped_cross_embedding = self_att + self.cross_attention_pos_embedding                        # embedd positional encoding for cross attention
            cross_att = self.cross_attention(reshaped_cross_embedding, c)
            s3 = cross_att.reshape(s3.shape)                                                                # reshape back to [B, H, W, C]

        ## Only self attention (optionally)
        elif self.is_self_attention:
            reshaped_s3 = s3.reshape(-1, s3.shape[1]*s3.shape[2], s3.shape[3])                              # reshape to [B, H*W, C]
            reshaped_self_embedding = reshaped_s3 + self.self_attention_pos_embedding                       # embedd positional encoding for self attention
            self_att = self.self_attention(reshaped_self_embedding, reshaped_self_embedding)
            s3 = self_att.reshape(s3.shape)                                                                 # reshape back to [B, H, W, C]

        ## Only cross attention (optionally)
        elif self.is_cross_attention:
            reshaped_s3 = s3.reshape(-1, s3.shape[1]*s3.shape[2], s3.shape[3])                              # reshape to [B, H*W, C]
            reshaped_cross_embedding = reshaped_s3 + self.cross_attention_pos_embedding                     # embedd positional encoding for cross attention
            cross_att = self.cross_attention(reshaped_cross_embedding, c)
            s3 = cross_att.reshape(s3.shape)                                                                # reshape back to [B, H, W, C]

        ## Add residual connection
        r = s3 + x                                                                                    # project input to match spatial size and channels   

        return r
    
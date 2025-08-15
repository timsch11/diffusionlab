from flax import nnx
import jax.numpy as jnp
from jax import Array
from diffusion.blocks.resnet import ResNet
from diffusion.blocks.pos_embedding import get_2d_sinusoidal_positional_encoding

class DownsampleBlock(nnx.Module):
    def __init__(self, height: int, width: int, in_channels: int, out_channels: int, timestamp_embedding_size: int, rngs: nnx.Rngs, self_attention: bool = False, self_attention_heads: int = 2, cross_attention: bool = False, cross_attention_heads: int = 2, text_embedding_dim: int = None, dtype: jnp.dtype = jnp.float32):
        super().__init__()

        # calculate downsampling factor
        if out_channels % in_channels != 0:
            raise ValueError(f"Invalid values for in_channels and out_channels, must evenly divide, values: in_channels: {in_channels}, out_channels: {out_channels}")
        
        self.downsampling_factor = out_channels // in_channels

        # conv layers (downssampling)
        self.conv1 = nnx.Conv(in_features=in_channels, out_features=out_channels, padding=1, strides=2, kernel_size=(3, 3), dtype=dtype, rngs=rngs)
        self.rngs = rngs

        # projection of timestamp embedding into channels
        self.timestamp_embedding_projection = nnx.Linear(in_features=timestamp_embedding_size, out_features=in_channels, dtype=dtype, rngs=rngs)

        # resnet blocks
        self.resnet1 = ResNet(in_channels, rngs=rngs, dtype=dtype)
        self.resnet2 = ResNet(in_channels, rngs=rngs, dtype=dtype)

        # whether to apply self/cross attention
        self.is_self_attention = self_attention
        self.is_cross_attention = cross_attention

        # initalize attention layers if neccessary
        # self attention
        if self_attention:
            # check for valid amount of attention heads
            if self_attention_heads < 1:
                raise ValueError("Invalid attention head parameter for self attention")
            
            self.self_attention_norm = nnx.GroupNorm(num_groups=24, num_features=in_channels, dtype=dtype, rngs=rngs)
            self.self_attention = nnx.MultiHeadAttention(in_features=in_channels, num_heads=self_attention_heads, qkv_features=in_channels, decode=False, dtype=dtype, rngs=rngs)
            self.self_attention_pos_embedding = get_2d_sinusoidal_positional_encoding(height, width, in_channels)
            #self.self_attention_pos_embedding = nnx.Param(value=jnp.full(shape=(1, height * width, in_channels), fill_value=0.0, dtype=dtype))

        # cross attention
        if cross_attention:
            # check for valid amount of attention heads
            if cross_attention_heads < 1:
                raise ValueError("Invalid attention head parameter for cross attention")
            
            if text_embedding_dim is None:
                raise ValueError("Set text embedding dimension if you want to use cross attention")

            self.cross_attention_norm = nnx.GroupNorm(num_groups=24, num_features=in_channels, dtype=dtype, rngs=rngs)
            self.cross_attention = nnx.MultiHeadAttention(in_features=in_channels, in_kv_features=text_embedding_dim, num_heads=cross_attention_heads, decode=False, dtype=dtype, rngs=rngs)
            self.cross_attention_pos_embedding = get_2d_sinusoidal_positional_encoding(height, width, in_channels)  # nnx.Param(value=jnp.full(shape=(1, height * width, in_channels), fill_value=0.0, dtype=dtype))

    @nnx.jit
    def __call__(self, x, t, c=None, msk=None) -> Array:
        ### Timestep embedding projection
        t_embedd = self.timestamp_embedding_projection(t)

        ### Residual Block
        # ResNet 2
        s = self.resnet1(x, t_embedd)

        # normalize mask to boolean and broadcastable shape [B, 1, 1, Tk]
        cross_mask = None
        if msk is not None:
            cross_mask = jnp.asarray(msk, dtype=jnp.bool_)
            if cross_mask.ndim == 2:
                cross_mask = cross_mask[:, None, None, :]
            elif cross_mask.ndim == 3:
                cross_mask = cross_mask[:, None, :, :]  # allow [B, Tq, Tk] -> [B, 1, Tq, Tk]

        ### Attention (optionally)
        # Combines self and cross attention if possible to enhance efficience
        if self.is_self_attention and self.is_cross_attention:
            reshaped_s = self.self_attention_norm(s.reshape(-1, s.shape[1]*s.shape[2], s.shape[3]))                              # reshape to [B, H*W, C]
            reshaped_self_embedding = reshaped_s + self.self_attention_pos_embedding[None, :, :]                       # embedd positional encoding for self attention
            self_att = self.cross_attention_norm(reshaped_s + self.self_attention(reshaped_self_embedding, reshaped_self_embedding))
            reshaped_cross_embedding = self_att + self.cross_attention_pos_embedding[None, :, :]                        # embedd positional encoding for cross attention
            cross_att = self_att + self.cross_attention(reshaped_cross_embedding, c, mask=cross_mask)
            s = cross_att.reshape(s.shape)                                                                # reshape back to [B, H, W, C]

        ## Only self attention (optionally)
        elif self.is_self_attention:
            reshaped_s = self.self_attention_norm(s.reshape(s.shape[0], s.shape[1]*s.shape[2], s.shape[3]))                              # reshape to [B, H*W, C]
            reshaped_self_embedding = reshaped_s + self.self_attention_pos_embedding[None, :, :]                       # embedd positional encoding for self attention
            self_att = reshaped_s + self.self_attention(reshaped_self_embedding, reshaped_self_embedding)
            s = self_att.reshape(s.shape)           
            #del self_att                                                                                   # reshape back to [B, H, W, C]

        ## Only cross attention (optionally)
        elif self.is_cross_attention:
            reshaped_s = self.cross_attention_norm(s.reshape(s.shape[0], s.shape[1]*s.shape[2], s.shape[3]))                              # reshape to [B, H*W, C]
            reshaped_cross_embedding = reshaped_s + self.cross_attention_pos_embedding[None, :, :]                     # embedd positional encoding for cross attention
            cross_att = reshaped_s + self.cross_attention(reshaped_cross_embedding, c, mask=cross_mask)
            s = cross_att.reshape(s.shape)                                                                # reshape back to [B, H, W, C]

        ## ResNet 2
        x_skip =  self.resnet2(s, t_embedd) 

        ### Downsampling
        x_downsampled = self.conv1(x_skip)

        return x_downsampled, x_skip
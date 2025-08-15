from flax import nnx
import jax.numpy as jnp
from jax import Array, image
from diffusion.blocks.resnet import ResNet
from diffusion.blocks.pos_embedding import get_2d_sinusoidal_positional_encoding


class UpsampleBlock(nnx.Module):
    def __init__(self, height: int, width: int, in_channels: int, out_channels: int, timestamp_embedding_size: int, rngs: nnx.Rngs, self_attention: bool = False, self_attention_heads: int = 2, cross_attention: bool = False, cross_attention_heads: int = 2, text_embedding_dim: int = None, dtype: jnp.dtype = jnp.float32):
        super().__init__()

        # calculate downsampling factor
        if in_channels % out_channels != 0:
            raise ValueError(f"Invalid values for in_channels and out_channels, must evenly divide, values: in_channels: {in_channels}, out_channels: {out_channels}")
        
        self.upsampling_factor = in_channels // out_channels

        # calc channels after concatenation of skip connection
        channels_after_skip_concat = in_channels   # out_channels*(self.upsampling_factor+1)

        # projection of timestamp embedding into channels
        self.timestamp_embedding_projection = nnx.Linear(in_features=timestamp_embedding_size, out_features=channels_after_skip_concat, dtype=dtype, rngs=rngs)

        # conv layers
        self.conv_skip = nnx.Conv(in_features=in_channels // 2, out_features=in_channels, kernel_size=1, dtype=dtype, rngs=rngs)
        self.conv1 = nnx.Conv(in_features=channels_after_skip_concat, out_features=out_channels, kernel_size=1, dtype=dtype, rngs=rngs)
        self.rngs = rngs

        # resnet blocks
        self.resnet1 = ResNet(channels_after_skip_concat, rngs=rngs, dtype=dtype)
        self.resnet2 = ResNet(channels_after_skip_concat, rngs=rngs, dtype=dtype)
        
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
            self.self_attention = nnx.MultiHeadAttention(in_features=channels_after_skip_concat, num_heads=self_attention_heads, qkv_features=channels_after_skip_concat, decode=False, dtype=dtype, rngs=rngs)
            self.self_attention_pos_embedding = get_2d_sinusoidal_positional_encoding(height * 2, width * 2, in_channels)
            # self.self_attention_pos_embedding = nnx.Param(value=jnp.full(shape=(1, height * width * 4, channels_after_skip_concat), fill_value=0.0, dtype=dtype))

        # cross attention
        if cross_attention:
            # check for valid amount of attention heads
            if cross_attention_heads < 1:
                raise ValueError("Invalid attention head parameter for cross attention")
            
            if text_embedding_dim is None:
                raise ValueError("Set text embedding dimension if you want to use cross attention")
            
            self.cross_attention_norm = nnx.GroupNorm(num_groups=24, num_features=in_channels, dtype=dtype, rngs=rngs)
            self.cross_attention = nnx.MultiHeadAttention(in_features=channels_after_skip_concat, in_kv_features=text_embedding_dim, num_heads=cross_attention_heads, decode=False, dtype=dtype, rngs=rngs)
            self.cross_attention_pos_embedding = get_2d_sinusoidal_positional_encoding(height * 2, width * 2, in_channels)
            # self.cross_attention_pos_embedding = nnx.Param(value=jnp.full(shape=(1, height * width * 4, channels_after_skip_concat), fill_value=0.0, dtype=dtype))

    @nnx.jit
    def __call__(self, x, x_skip, t, c=None, msk=None) -> Array:

        ### Upsampling (nearest neighbor)
        x = image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method="nearest")

        # normalize mask to boolean and broadcastable shape [B, 1, 1, Tk]
        cross_mask = None
        if msk is not None:
            cross_mask = jnp.asarray(msk, dtype=jnp.bool_)
            if cross_mask.ndim == 2:
                cross_mask = cross_mask[:, None, None, :]
            elif cross_mask.ndim == 3:
                cross_mask = cross_mask[:, None, :, :]  # allow [B, Tq, Tk] -> [B, 1, Tq, Tk]

        ### Concat skip connection
        #x = jnp.concatenate((x, x_skip), axis=-1)
        x = x + self.conv_skip(x_skip)

        ## timestep embedding projection
        t_embedd = self.timestamp_embedding_projection(t)

        ### ResNet 1
        s = self.resnet1(x, t_embedd)
                  
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
            s = cross_att.reshape(s.shape)                                                                # reshape back to [B, H, W, C]                                                                # reshape back to [B, H, W, C]

        ### ResNet 2
        r = self.resnet2(s, t_embedd) 
    
        ## Conv
        return self.conv1(r)                                                                                # reduce channels to out_channels
    

class UpsampleBlockAtt(nnx.Module):
    """
    Upsamples a given feature map by factor 2.
    Uses cross attention to incorporate skip connections.

    """

    def __init__(self, height: int, width: int, in_channels: int, out_channels: int, timestamp_embedding_size: int, rngs: nnx.Rngs, self_attention: bool = False, self_attention_heads: int = 2, cross_attention: bool = False, cross_attention_heads: int = 2, text_embedding_dim: int = None, dtype: jnp.dtype = jnp.float32):
        super().__init__()

        # calculate downsampling factor
        if in_channels % out_channels != 0:
            raise ValueError(f"Invalid values for in_channels and out_channels, must evenly divide, values: in_channels: {in_channels}, out_channels: {out_channels}")
        
        self.upsampling_factor = in_channels // out_channels

        # projection of timestamp embedding into channels
        self.timestamp_embedding_projection = nnx.Linear(in_features=timestamp_embedding_size, out_features=out_channels, dtype=dtype, rngs=rngs)

        # conv layers
        self.conv1 = nnx.Conv(in_features=out_channels, out_features=out_channels, padding=1, kernel_size=(3, 3), dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=out_channels, out_features=out_channels, padding=1, kernel_size=(3, 3), dtype=dtype, rngs=rngs)
        self.rngs = rngs

        # norm layers
        #self.norm1 = nnx.LayerNorm(num_features=out_channels, feature_axes=-1, dtype=dtype, rngs=rngs)
        norm1_groups = 1 if out_channels < 30 else 30
        self.norm1 = nnx.GroupNorm(num_groups=norm1_groups, num_features=out_channels, dtype=dtype, rngs=rngs)
        
        # whether to apply self/cross attention
        self.is_self_attention = self_attention
        self.is_cross_attention = cross_attention

        self.out_features = out_channels

        # cross attention for incorporating the skip connections
        self.cross_attention_skip = nnx.MultiHeadAttention(in_features=in_channels, in_kv_features=out_channels, out_features=out_channels, num_heads=3, decode=False, dtype=dtype, rngs=rngs)
        self.cross_attention_pos_embedding_inp_skip = nnx.Param(value=jnp.full(shape=(1, height * width * 4, in_channels), fill_value=0.0, dtype=dtype))
        self.cross_attention_pos_embedding_skip_skip = nnx.Param(value=jnp.full(shape=(1, height * width * 4, out_channels), fill_value=0.0, dtype=dtype))

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

        ### Concat skip connection via cross attention
        x_skip_reshaped = x_skip.reshape(-1, x_skip.shape[1]*x_skip.shape[2], x_skip.shape[3])             # merge H and W 
        x_reshaped = x.reshape(-1, x.shape[1]*x.shape[2], x.shape[3])

        x_skip_embedd = x_skip_reshaped + self.cross_attention_pos_embedding_skip_skip      # add positional encoding
        x_embedd = x_reshaped + self.cross_attention_pos_embedding_inp_skip

        x_cross_att = self.cross_attention_skip(x_embedd, x_skip_embedd)                    # compute cross attention

        x = x_cross_att.reshape(x.shape[0], x.shape[1], x.shape[2], self.out_features)

        ### Residual Block
        ## Conv
        x = self.conv1(x)                                                           # conv 1
        s2 = x + self.timestamp_embedding_projection(t)[:, None, None, :]           # project timestamp embedding
        s = self.conv2(nnx.silu(self.norm1(s2)))                                   # conv 2             

        ## Attention (optionally)
        # Combines self and cross attention if possible to enhance efficience
        if self.is_self_attention and self.is_cross_attention:
            reshaped_s = s.reshape(-1, s.shape[1]*s.shape[2], s.shape[3])                              # reshape to [B, H*W, C]
            reshaped_self_embedding = reshaped_s + self.self_attention_pos_embedding                       # embedd positional encoding for self attention
            self_att = self.self_attention(reshaped_self_embedding, reshaped_self_embedding)
            reshaped_cross_embedding = self_att + self.cross_attention_pos_embedding                        # embedd positional encoding for cross attention
            cross_att = self.cross_attention(reshaped_cross_embedding, c)
            s = cross_att.reshape(s.shape)                                                                # reshape back to [B, H, W, C]

        ## Only self attention (optionally)
        elif self.is_self_attention:
            reshaped_s = s.reshape(-1, s.shape[1]*s.shape[2], s.shape[3])                              # reshape to [B, H*W, C]
            reshaped_self_embedding = reshaped_s + self.self_attention_pos_embedding                       # embedd positional encoding for self attention
            self_att = self.self_attention(reshaped_self_embedding, reshaped_self_embedding)
            s = self_att.reshape(s.shape)                                                                 # reshape back to [B, H, W, C]

        ## Only cross attention (optionally)
        elif self.is_cross_attention:
            reshaped_s = s.reshape(-1, s.shape[1]*s.shape[2], s.shape[3])                              # reshape to [B, H*W, C]
            reshaped_cross_embedding = reshaped_s + self.cross_attention_pos_embedding                     # embedd positional encoding for cross attention
            cross_att = self.cross_attention(reshaped_cross_embedding, c)
            s = cross_att.reshape(s.shape)                                                                # reshape back to [B, H, W, C]

        ## Add residual connection
        r = s + x                                                                                    # project input to match spatial size and channels   

        return r
    
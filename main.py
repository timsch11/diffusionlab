from diffusion.forward import apply_noise
from diffusion.model import DiffusionNet, DownsampleBlock, UpsampleBlock, get_timestep_embedding
from util import load_image, save_image, rescale_image

import jax.numpy as jnp
from jax import random, profiler, image
from flax import nnx
from tqdm import tqdm
import jax


T = 100
T_dim = 100
H = 64
W = 64

img = rescale_image(target_height=H, target_width=W, img_path="coolQuantPC.jpg", normalize=True)
#tsave_image("downscaled_image.jpeg", rimg)

img = img.reshape(1, *img.shape)


img = jnp.concatenate((img, img, img, img), axis=0)

"""tembedd = get_timestep_embedding(jnp.array([T]), dim=T_dim)
e1 = DownsampleBlock(height=H, width=W, in_channels=3, out_channels=6, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=False)

H = -(H // -2)
W = -(W // -2)

model = UpsampleBlock(H, W, 6, 3, T_dim, rngs=nnx.Rngs(params=random.key(32)))

r = e1(img, tembedd)
u = model(r, img, tembedd)

print(u.shape)"""


tembedd = get_timestep_embedding(jnp.array([T]), dim=T_dim)
e1 = DownsampleBlock(height=H, width=W, in_channels=3, out_channels=6, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=False)
H = -(H // -2)
W = -(W // -2)
e2 = DownsampleBlock(height=H, width=W, in_channels=6, out_channels=12, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=False)
H = -(H // -2)
W = -(W // -2)
e3 = DownsampleBlock(height=H, width=W, in_channels=12, out_channels=24, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=True)
H = -(H // -2)
W = -(W // -2)
e4 = DownsampleBlock(height=H, width=W, in_channels=24, out_channels=48, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=True)
H = -(H // -2)
W = -(W // -2)
e5 = DownsampleBlock(height=H, width=W, in_channels=48, out_channels=96, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=True)
H = -(H // -2)
W = -(W // -2)
e6 = DownsampleBlock(height=H, width=W, in_channels=96, out_channels=192, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=True)
H = -(H // -2)
W = -(W // -2)
e7 = DownsampleBlock(height=H, width=W, in_channels=192, out_channels=384, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=True)
H = -(H // -2)
W = -(W // -2)
e8 = DownsampleBlock(height=H, width=W, in_channels=384, out_channels=768, timestamp_embedding_size=T_dim, rngs=nnx.Rngs(params=random.key(32)), self_attention=True)

for i in tqdm(range(100)):
    r = e1(img, tembedd)
    r2 = e2(r, tembedd)
    r3 = e3(r2, tembedd)
    r4 = e4(r3, tembedd)
    r5 = e5(r4, tembedd)
    r6 = e6(r5, tembedd)
    r7 = e7(r6, tembedd)
    r8 = e8(r7, tembedd)

print(r8.shape)

# mod_img = apply_noise(img, T, betas = jnp.linspace(1e-4, 0.02, T))
# save_image("output.jpeg", mod_img)
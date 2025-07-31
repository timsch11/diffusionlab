from diffusion.forward import apply_noise
from diffusion.model import DiffusionNet, DownsampleBlock, get_timestep_embedding
from util import load_image, save_image

import jax.numpy as jnp
from jax import random
from flax import nnx



T = 100

img = load_image(img_path="coolQuantPC.jpg", normalize=True)
"""print(img.shape)
tembedd = get_timestep_embedding(jnp.array([T]), dim=100)
model = DownsampleBlock(inp_shape=img.shape, in_features=1, out_features=2, timestamp_embedding_size=100, rngs=nnx.Rngs(params=random.key(32)))

print(model(img, tembedd))"""
mod_img = apply_noise(img, T, betas = jnp.linspace(1e-4, 0.02, T))
save_image("output.jpeg", mod_img)
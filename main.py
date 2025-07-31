from diffusion.forward import apply_noise
from util import load_image, save_image

import jax.numpy as jnp


T = 100

img = load_image(img_path="coolQuantPC.jpg", normalize=True)
mod_img = apply_noise(img, T, betas = jnp.linspace(1e-4, 0.02, T))
save_image("output.jpeg", mod_img)
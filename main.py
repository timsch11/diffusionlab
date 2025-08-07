# import os
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


from diffusion.forward import apply_t_noise_steps, apply_noise_step
from diffusion.model import DiffusionNet
from util import load_image, save_image, rescale_image
from prompt_embedding import embedd_prompts_batched

import jax.numpy as jnp
from jax import random, profiler, image
from flax import nnx
from tqdm import tqdm
import jax
import optax
from schedule import cosine_beta_schedule

from util import load_model, save_model

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_hidden, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM

import time


img = rescale_image(target_height=H, target_width=W, img_path="emojiimage-dataset/image/Google/2.png", normalize=True, dtype=DTYPE)
#img = load_image(img_path="coolQuantPC.jpg", normalize=True, dtype=DTYPE)

schedule = cosine_beta_schedule(T)

T = 178

img = apply_t_noise_steps(img, T, betas = schedule[:T], dtype=DTYPE)
noise_img = apply_noise_step(img, schedule[T], dtype=DTYPE)

embeddings = embedd_prompts_batched(["grinning face with big eyes"])


# noise_img = noise_img.reshape(1, *noise_img.shape)
# img = img.reshape(1, *img.shape)

img = jnp.stack([img])
noise_img = jnp.stack([noise_img])

model_x = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_hidden=T_hidden, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)

model = load_model(model_x, "models_v5/final")

r = model(noise_img, jnp.array([T+1]), embeddings)

print(jnp.mean(jnp.abs(r - img)))

"""
for t in tqdm(range(T, 0, -1)):
    noise_img = model(noise_img, jnp.array([t]), embeddings)

    # Clip the output to the valid training range [-1, 1]
    noise_img = jnp.clip(noise_img, -1.0, 1.0)

print(noise_img.dtype)

img_squeezed = jnp.squeeze(noise_img, axis=0)
img_rescaled = (img_squeezed + 1) / 2.0

save_image("output_test.jpg", img_rescaled)"""
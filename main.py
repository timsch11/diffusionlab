# import os
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


from diffusion.forward import apply_t_noise_steps, apply_noise_step
from diffusion.model import DiffusionNet
from util import load_image, save_image, rescale_image

import jax.numpy as jnp
from jax import random, profiler, image
from flax import nnx
from tqdm import tqdm
import jax
import optax
from schedule import cosine_beta_schedule

from util import load_model, save_model


DTYPE = jnp.float32

B = 4
T = 200

T_dim = 128
T_hidden = 1024
T_out = 128

H = 64
W = 64

import time


img = rescale_image(target_height=H, target_width=W, img_path="coolQuantPC.jpg", normalize=True, dtype=DTYPE)
#img = load_image(img_path="coolQuantPC.jpg", normalize=True, dtype=DTYPE)

schedule = cosine_beta_schedule(T)

noise_img = apply_t_noise_steps(img, 1, betas = schedule[:1], dtype=DTYPE)
embed = jnp.full(shape=(1, 384), fill_value=1.32)


# noise_img = noise_img.reshape(1, *noise_img.shape)
# img = img.reshape(1, *img.shape)

img = jnp.stack([img for _ in range(B)])
noise_img = jnp.stack([noise_img for _ in range(B)])
embeddings = jnp.stack([embed for _ in range(B)])

model_x = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=4, t_in=T_dim, t_hidden=T_hidden, t_out=T_out, dtype=DTYPE, text_embedding_dim=384, rngs=nnx.Rngs(params=random.key(32)))

params = nnx.state(model_x, nnx.Param)


save_model(model_x, "models")

model = load_model(model_x, "/home/ts/Desktop/projects/diffusionlab/models")

total_params = 0
for x in jax.tree_util.tree_leaves(params):
    r = 1
    for p_dim in x.shape:
        r *= p_dim

    total_params += r


print("Total parameters of model: ", total_params)  # 10.738.835


optimizer = optax.adam(0.0001)
opt_state = optimizer.init(params)


def loss_fn(model, x, c, y, t):
    t_array = jnp.full((x.shape[0],), t, dtype=DTYPE)
    model_output = model(x, t_array, c)
    loss = jnp.mean((model_output - y) ** 2, dtype=DTYPE)
    return loss


loss_fn_functional = nnx.jit(nnx.value_and_grad(loss_fn))

for _ in range(100):
    for __ in tqdm(range(10)):
        print("here")
        loss, grads = loss_fn_functional(model, noise_img, embeddings, img, T)
        
        print("here2")

        # Extract parameters and apply updates
        params = nnx.state(model, nnx.Param)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Update the model with new parameters
        nnx.update(model, new_params)
        
    print(loss)

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

DTYPE = jnp.float32

B = 4
T = 200

T_dim = 128
T_hidden = 1024
T_out = 128

H = 128
W = 128

import time


#img = rescale_image(target_height=H, target_width=W, img_path="coolQuantPC.jpg", normalize=True, dtype=DTYPE)
img = load_image(img_path="coolQuantPC.jpg", normalize=True, dtype=DTYPE)


noise_img = apply_t_noise_steps(img, 0, betas = cosine_beta_schedule(T)[:0], dtype=DTYPE)
save_image("output4.jpeg", noise_img)

exit(0)


# noise_img = noise_img.reshape(1, *noise_img.shape)
# img = img.reshape(1, *img.shape)

img = jnp.stack([img for _ in range(B)])
noise_img = jnp.stack([noise_img for _ in range(B)])

model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=4, t_in=T_dim, t_hidden=T_hidden, t_out=T_out, dtype=DTYPE, rngs=nnx.Rngs(params=random.key(32)))

params = nnx.state(model, nnx.Param)


total_params = 0
for x in jax.tree_util.tree_leaves(params):
    r = 1
    for p_dim in x.shape:
        r *= p_dim

    total_params += r


print("Total parameters of model: ", total_params)  # 10.738.835


optimizer = optax.adam(0.0001)
opt_state = optimizer.init(params)


def loss_fn(model, x, y, t):
    t_array = jnp.full((x.shape[0],), t, dtype=DTYPE)
    model_output = model(x, t_array)
    loss = jnp.mean((model_output - y) ** 2, dtype=DTYPE)
    return loss


loss_fn_functional = nnx.jit(nnx.value_and_grad(loss_fn), donate_argnames=("x", "y", "t"))

for _ in range(100):
    for __ in tqdm(range(10)):
        loss, grads = loss_fn_functional(model, noise_img, img, T)
        
        # Extract parameters and apply updates
        params = nnx.state(model, nnx.Param)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Update the model with new parameters
        nnx.update(model, new_params)
        
    print(loss)

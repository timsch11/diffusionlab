from diffusion.forward import apply_noise
from diffusion.model import DiffusionNet
from util import load_image, save_image, rescale_image

import jax.numpy as jnp
from jax import random, profiler, image
from flax import nnx
from tqdm import tqdm
import jax
import optax


B = 1
T = 1

T_dim = 128
T_hidden = 1024
T_out = 128

H = 128
W = 128

img = rescale_image(target_height=H, target_width=W, img_path="coolQuantPC.jpg", normalize=True)
noise_img = apply_noise(img, T, betas = jnp.linspace(1e-4, 0.02, T))

# noise_img = noise_img.reshape(1, *noise_img.shape)
# img = img.reshape(1, *img.shape)

img = jnp.stack([img for _ in range(B)])
noise_img = jnp.stack([noise_img for _ in range(B)])

print(img.shape)

model = DiffusionNet(height=H, width=W, channels=3, t_in=T_dim, t_hidden=T_hidden, t_out=T_out, rngs=nnx.Rngs(params=random.key(32)))

params = nnx.state(model, nnx.Param) 

total_params = 0
for x in jax.tree_util.tree_leaves(params):
    r = 1
    for p_dim in x.shape:
        r *= p_dim

    total_params += r


print("Total parameters of model: ", total_params)

exit(0)

optimizer = optax.adam(0.01)
opt_state = optimizer.init(params)


def loss_fn(model, x, y):
    model_output = model(x, T)
    loss = jnp.mean((model_output - y) ** 2)
    return loss

# Create a functional version of the loss for gradient computation
loss_fn_functional = nnx.jit(nnx.value_and_grad(loss_fn))

for _ in range(100):
    for __ in tqdm(range(10)):
        loss, grads = loss_fn_functional(model, noise_img, img)
        
        # Extract parameters and apply updates
        params = nnx.state(model, nnx.Param)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Update the model with new parameters
        nnx.update(model, new_params)
        
    print(loss)


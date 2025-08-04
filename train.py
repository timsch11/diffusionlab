# import os
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


from diffusion.model import DiffusionNet
from schedule import cosine_beta_schedule
from dataloader import Dataloader

from util import save_model, load_model

import jax.numpy as jnp
from jax import random
from flax import nnx
from tqdm import tqdm
import jax
import optax


DTYPE = jnp.float32

B = 8
EPOCHS = 2

T = 200

T_dim = 128
T_hidden = 1024
T_out = 128

TEXT_EMBEDDING_DIM = 384

H = 64
W = 64

CHANNEL_SAMPLING_FACTOR = 4
RNGS = nnx.Rngs(params=random.key(32))


SCHEDULE = cosine_beta_schedule(T)

model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, t_in=T_dim, t_hidden=T_hidden, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)

print("Model initalized successfully")

params = nnx.state(model, nnx.Param)


save_model(model, "models/test")

model2 = load_model(model, "models/test")

exit()


### print amount of params
total_params = 0
for x in jax.tree_util.tree_leaves(params):
    r = 1
    for p_dim in x.shape:
        r *= p_dim

    total_params += r


print("Total parameters of model: ", total_params)  # 10.738.835


### Training

optimizer = optax.adam(0.0001)
opt_state = optimizer.init(params)


def loss_fn(model, x, t, c, y):
    t_array = jnp.full((x.shape[0],), t, dtype=DTYPE)
    model_output = model(x, t_array, c)
    loss = jnp.mean((model_output - y) ** 2, dtype=DTYPE)
    return loss


loss_fn_jitted = nnx.jit(nnx.value_and_grad(loss_fn))

print("Initalizing dataloader...")
dataloader = Dataloader(data_dir="emojiimage-dataset/image/Google", csv_file_path="emojiimage-dataset/full_emoji.csv", target_height=H, target_width=W, timesteps=T, schedule=SCHEDULE, batch_size=4, dtype=jnp.float32)
print("Dataloader successfully initalized")


for epoch in range(EPOCHS):
    loss = jnp.array([0])
    for x, t, c, y in tqdm(dataloader):
        new_loss, grads = loss_fn_jitted(model, x, t, c, y)
        
        # Extract parameters and apply updates
        params = nnx.state(model, nnx.Param)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Update the model with new parameters
        nnx.update(model, new_params)

        loss += new_loss
        
    loss /= (1816 / B)
    print(f"Loss after epoch {epoch}: {loss}")


# Save state
params = nnx.state(model, nnx.Param)
nnx.save(params, "model_state.msgpack")
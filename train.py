import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"  # let jax preallocate 90% of available vram -> increases efficiency


from diffusion.model import DiffusionNet
from dataloader import Dataloader
from pipeline import DiffusionPipeline
from prompt_embedding import embedd_prompts_seq

from util import save_model, load_model, save_image

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM, RANDOMKEY, MAX_INDEX

import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm
import jax
import optax


### Model initalization
model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)
print("Model initalized successfully")
# model = load_model(model, "models_vA0/epoch150")

### Print count of params
params = nnx.state(model, nnx.Param)

total_params = 0
for x in jax.tree_util.tree_leaves(params):
    r = 1
    for p_dim in x.shape:
        r *= p_dim

    total_params += r

print("Total parameters of model: ", total_params)  # 17.641.739

### Training

# lr schedule params
warmup_epochs = 3
init_lr = 5e-7
peak_lr = 4e-4
end_lr = 5e-7
steps_per_epoch = 1  # MAX_INDEX // B

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=init_lr,
    peak_value=peak_lr,
    warmup_steps=warmup_epochs * steps_per_epoch,
    decay_steps=(EPOCHS - warmup_epochs) * steps_per_epoch,
    end_value=end_lr
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients to a max norm of 1.0
    optax.adam(learning_rate=lr_schedule)
)

opt_state = optimizer.init(params)


### Loss functions
def mse(model, x, t, c, msk, y):
    model_output = model(x, t, c, msk)
    loss = jnp.mean((model_output - y) ** 2, dtype=DTYPE)
    return loss


def fmse(model, x, t, c, msk, y):
    model_output = model(x, t, c, msk)
    loss = jnp.mean((10 * (model_output - y)) ** 2, dtype=DTYPE)
    return loss


def mae(model, x, t, c, msk, y):
    model_output = model(x, t, c, msk)
    loss = jnp.mean(jnp.abs(model_output - y), dtype=DTYPE)
    return loss

loss_fn_jitted = nnx.jit(nnx.value_and_grad(mse))


# Test model with imagegen pipeline for later evaluation
pipe = DiffusionPipeline(H, W, model, embedd_prompts_seq, 384, T, SCHEDULE)
pipe.generate_image("Grinning face", "validation_images/test.jpeg")


### Init dataloader
print("Initalizing dataloader...")
dataloader = Dataloader(data_dir="emojiimage-dataset/image/Google", csv_file_path="emojiimage-dataset/full_emoji.csv", target_height=H, target_width=W, embedding_dim = 384, embedding_dropout=0.1, timesteps=T, schedule=SCHEDULE, batch_size=B, dtype=jnp.float32, key=RANDOMKEY, max_index=MAX_INDEX)
print("Dataloader successfully initalized")

num_batches = -(dataloader.num_items // -B)


### Training loop
for epoch in range(EPOCHS): 
    loss = jnp.array([0])
    for x, t, c, msk, y in tqdm(dataloader, total=num_batches):
        new_loss, grads = loss_fn_jitted(model, x, t, c, msk, y)
        
        # Extract parameters and apply updates
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Update the model with new parameters
        nnx.update(model, params)

        loss += new_loss
        
    loss /= num_batches
    print(f"Loss after epoch {epoch}: {loss}")

    if (epoch) % 25 == 0:
        save_model(model, f"models_vA0/epoch{epoch}")

        pipe.model = model
        pipe.generate_image("Grinning face", f"validation_images/epoch{epoch}.jpeg")


### Save state
save_model(model, "models_vA0/final")  #17.772.891
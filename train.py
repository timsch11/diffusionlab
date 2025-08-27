import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  # let jax preallocate 90% of available vram -> increases efficiency


from model import DiffusionNet
from dataloader import Dataloader
from diffusion.pipeline import DiffusionPipeline
from diffusion.prompt_embedding import embedd_prompts_seq

from util import save_model, load_model

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM, RANDOMKEY, MAX_INDEX, DATASET_MEASURE_FILE

import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm
import jax
import optax


### Model initalization
model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)
print("Model initalized successfully")

### Print count of params
params = nnx.state(model, nnx.Param)

total_params = 0
for x in jax.tree_util.tree_leaves(params):
    r = 1
    for p_dim in x.shape:
        r *= p_dim

    total_params += r

print("Total parameters of model: ", total_params)  # 13.915.087

### Init dataloader
print("Initalizing dataloader...")
dataloader = Dataloader(data_dir="emojiimage-dataset/image/JoyPixels", csv_file_path="emojiimage-dataset/full_emoji.csv", target_height=H, target_width=W, embedding_dim = TEXT_EMBEDDING_DIM, embedding_dropout=0.1, timesteps=T, schedule=SCHEDULE, batch_size=B, dtype=jnp.float32, key=RANDOMKEY, max_index=MAX_INDEX, file_storage=DATASET_MEASURE_FILE)
dataloader.epoch = 395
print("Dataloader successfully initalized")

num_batches = -(dataloader.num_items // -B)

### Training
#num_examples = dataloader.num_items
total_steps = EPOCHS * num_batches
warmup_steps = 500
decay_steps  = total_steps - warmup_steps

init_lr  = 2e-6
peak_lr  = 6e-5
final_lr = 2e-6    

lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=init_lr,
    peak_value=peak_lr,
    warmup_steps=warmup_steps,
    decay_steps=decay_steps,
    end_value=final_lr,
)

# ---- weight decay mask: decay only parameters with ndim > 1 ----
def decay_mask(tree):
    return jax.tree.map(lambda p: getattr(p, "ndim", 0) > 1, tree)

# AdamW = add_decayed_weights BEFORE Adam (decoupled weight decay)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.masked(optax.add_decayed_weights(1e-4), decay_mask),
    optax.adam(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8),
)

optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)

# split before training loop
graphdef, state = nnx.split((model, optimizer, metrics))

@jax.jit
def train_step(graphdef, state, x, t, c, msk, y):
  model, optimizer, metrics = nnx.merge(graphdef, state)

  def loss_fn(model):
    y_pred = model(x, t, c, msk)
    return ((y_pred - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)
  metrics.update(loss=loss)

  state = nnx.state((model, optimizer, metrics))
  return loss, state


# Test model with imagegen pipeline for later evaluation
pipe = DiffusionPipeline(H, W, model, embedd_prompts_seq, TEXT_EMBEDDING_DIM, T, SCHEDULE, DATASET_MEASURE_FILE)

### Training loop
for epoch in range(EPOCHS): 
    loss = jnp.array([0])
    for x, t, c, msk, y in tqdm(dataloader, total=num_batches):
        new_loss, state = train_step(graphdef, state, x, t, c, msk, y)

        loss += new_loss
        
    loss /= num_batches
    print(f"Loss after epoch {epoch}: {loss}")

    if (epoch) % 25 == 0:
        # update objects after training
        nnx.update((model, optimizer, metrics), state)

        pipe.model = model
        pipe.generate_images("cat", "yellow heart", "collision", "happy face", "wood", "airplane", target_directory=f"validation_images/epoch{epoch}/", cfg=True)

        save_model(model, f"model_medium/epoch{epoch}")

# update objects after training
nnx.update((model, optimizer, metrics), state)

### Save state
save_model(model, "model_medium/final")

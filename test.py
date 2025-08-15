from diffusion.model import DiffusionNet
from schedule import cosine_beta_schedule
from dataloader import Dataloader
from pipeline import DiffusionPipeline, DiffusionPipeline

from prompt_embedding import embedd_prompts_batched, embedd_prompts_seq

from util import save_model, load_model
import orbax.checkpoint as orbax

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_hidden, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM

import jax.numpy as jnp
from jax import random
from flax import nnx
from tqdm import tqdm
import jax
import optax


model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)

state = nnx.state(model)

# Load the parameters
checkpointer = orbax.PyTreeCheckpointer()
state = checkpointer.restore("/home/ts/Desktop/projects/diffusionlab/diffusionlab/models_vA0/final", item=state)

# update the model with the loaded state
nnx.update(model, state)

pipeline = DiffusionPipeline(H, W, model, embedd_prompts_seq, 384, T, SCHEDULE)


pipeline("cold face", "cold.jpeg")
pipeline("Grinning face", "grinning.jpeg")
pipeline("glasses", "glasses.jpeg")
pipeline("greece", "greece.jpeg")
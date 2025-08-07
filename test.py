from diffusion.model import DiffusionNet
from schedule import cosine_beta_schedule
from dataloader import Dataloader
from pipeline import DiffusionPipeline

from prompt_embedding import embedd_prompts_batched

from util import save_model, load_model

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_hidden, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM

import jax.numpy as jnp
from jax import random
from flax import nnx
from tqdm import tqdm
import jax
import optax


model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_hidden=T_hidden, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)
pipeline = DiffusionPipeline(H, W, "models_v5/final", model, embedd_prompts_batched, 200, SCHEDULE)


test = "Smiling face"
pipeline(test, "output_v3.jpeg")
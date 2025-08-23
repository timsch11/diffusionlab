from diffusion.model import DiffusionNet
from pipeline import DiffusionPipeline, DiffusionPipeline

from prompt_embedding import embedd_prompts_seq

import orbax.checkpoint as orbax

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM, DATASET_MEASURE_FILE

from flax import nnx


model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)

state = nnx.state(model)

# Load the parameters
checkpointer = orbax.PyTreeCheckpointer()
state = checkpointer.restore("/home/ts/Desktop/projects/diffusionlab/diffusionlab/model_clip_std/epoch849", item=state)

# update the model with the loaded state
nnx.update(model, state)

pipeline = DiffusionPipeline(H, W, model, embedd_prompts_seq, TEXT_EMBEDDING_DIM, T, SCHEDULE, DATASET_MEASURE_FILE)


pipeline("cat", "cat.jpeg", cfg=True)
pipeline("sweat", "sweat.jpeg", cfg=True)
exit(0)
pipeline("cold face", "cold.jpeg", cfg=True)
pipeline("Grinning face", "grinning.jpeg", cfg=True)

"""
pipeline("cold face", "cold2.jpeg", cfg=False)
pipeline("Grinning face", "grinning2.jpeg", cfg=False)"""
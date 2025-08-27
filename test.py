from model import DiffusionNet
from diffusion.pipeline import DiffusionPipeline, DiffusionPipeline

from diffusion.prompt_embedding import embedd_prompts_seq

import orbax.checkpoint as orbax

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM, DATASET_MEASURE_FILE

from flax import nnx


model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)

state = nnx.state(model)

# Load the parameters
checkpointer = orbax.PyTreeCheckpointer()
state = checkpointer.restore("model_small", item=state)

# update the model with the loaded state
nnx.update(model, state)


ts = [250, 200, 150, 100, 50, 20, 10]

pipeline = DiffusionPipeline(H, W, model, embedd_prompts_seq, TEXT_EMBEDDING_DIM, T, SCHEDULE, DATASET_MEASURE_FILE)


for t in ts:
    pipeline.num_timesteps = t
    pipeline.generate_images("baby with brown hair", target_directory=f"exampleimgs_small/t{t}/", cfg=True)

"""pipeline("cat", "m2img/cat.jpeg", cfg=True)
pipeline("sweat", "m2img/sweat.jpeg", cfg=True)
pipeline("cold face", "m2img/cold.jpeg", cfg=True)
pipeline("Grinning face", "m2img/grinning.jpeg", cfg=True)
pipeline("Pink heart", "m2img/pink_heart.jpeg", cfg=True)"""

"""
pipeline("cold face", "cold2.jpeg", cfg=False)
pipeline("Grinning face", "grinning2.jpeg", cfg=False)"""
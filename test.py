from model import DiffusionNet
from diffusion.pipeline import DiffusionPipeline, DiffusionPipeline

from diffusion.prompt_embedding import embedd_prompts_seq

from params import B, CHANNEL_SAMPLING_FACTOR, DTYPE, EPOCHS, H, W, RNGS, SCHEDULE, T_dim, T_out, T, TEXT_EMBEDDING_DIM, BASE_DIM, DATASET_MEASURE_FILE
from util import load_model


# declare template model 
model = DiffusionNet(height=H, width=W, channels=3, channel_sampling_factor=CHANNEL_SAMPLING_FACTOR, base_dim=BASE_DIM, t_in=T_dim, t_out=T_out, text_embedding_dim=TEXT_EMBEDDING_DIM, dtype=DTYPE, rngs=RNGS)

# load params into model
model = load_model(model, "model/model_small")

# setup pipeline
pipeline = DiffusionPipeline(H, W, model, embedd_prompts_seq, TEXT_EMBEDDING_DIM, T, SCHEDULE, DATASET_MEASURE_FILE)


# generate image
PROMPT = "Cat"  # change
pipeline.generate_image(PROMPT, output_path=f"your_images/{PROMPT}.jpeg")

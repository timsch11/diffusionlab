from random import randint

import jax.numpy as jnp
from jax import Array, random
from prompt_embedding import embedd_prompts_batched
from util import load_model, save_image
from flax.nnx import Module, Rngs

from tqdm import tqdm


class DiffusionPipeline:
    def __init__(self, height: int, width: int, model_path: str, template_model: Module, text_embedding_function, timesteps: int, noise_schedule: Array):
        self.model = load_model(model_without_params=template_model, path=model_path)
        self.embedd = text_embedding_function
        self.schedule = noise_schedule
        self.timesteps = timesteps

        self.height = height
        self.width = width

    def generate_image(self, text_prompt: str, output_path: str):
        # embedd text 
        context = self.embedd([text_prompt])

        # generate noise
        img = random.normal(key=random.key(randint(1, 100000000)), shape=(1, self.height, self.width, 3))

        # refine noise over t timesteps
        for t in tqdm(range(self.timesteps, 0, -1)):
            img = self.model(img, jnp.array([t]), context)
            #img = jnp.clip(img, min=-1, max=1)

        # Remove batch dimension
        img_squeezed = jnp.squeeze(img, axis=0)

        print(img_squeezed)

        # Normalize
        img_rescaled = (img_squeezed + 1) / 2.0

        save_image(img_path=output_path, img=img_rescaled)
        print(f"Image saved to {output_path}")

    def __call__(self, text_prompt: str, output_path: str):
        self.generate_image(text_prompt, output_path)
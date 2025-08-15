from random import randint

import jax.numpy as jnp
from jax import Array, random
from prompt_embedding import embedd_prompts_batched
from util import load_model, save_image
from flax.nnx import Module, Rngs

from tqdm import tqdm


class DiffusionPipeline:
    """
    DiffusionPipeline(height, width, model, text_embedding_function, embedding_dim, num_timesteps, noise_schedule)
    A minimal diffusion sampling pipeline that generates images from text prompts using a provided model
    and text embedding function. The pipeline performs deterministic reverse diffusion (no added noise
    during sampling) from Gaussian noise to an image, supports optional classifier-free guidance (CFG),
    and writes the resulting image to disk.
    Parameters
    ----------
    height : int
        Height of the output image in pixels.
    width : int
        Width of the output image in pixels.
    model : callable
        A function or object implementing the trained diffusion model. Expected call signature:
        model(image: jnp.ndarray, t: jnp.ndarray, context: jnp.ndarray, att_mask: jnp.ndarray) -> jnp.ndarray,
        where `image` has shape (B, H, W, C), `t` is a 1-D array with the current timestep, and `context`
        and `att_mask` are the token embedding and attention mask returned by `text_embedding_function`.
    text_embedding_function : callable
        Function to encode text prompts into embeddings. Expected call signature:
        text_embedding_function([prompt_str]) -> (context: jnp.ndarray, att_mask: jnp.ndarray).
        `context` shape should be (1, N, embedding_dim).
    embedding_dim : int
        Dimensionality of the token embeddings produced by `text_embedding_function`.
    num_timesteps : int
        Number of discrete diffusion timesteps (T). The reverse loop runs from T down to 1.
    noise_schedule : jnp.ndarray
        A 1-D array of betas (noise schedule) of shape (T,). The pipeline computes alphas = 1 - betas
        and cumulative products of alphas for closed-form denoising steps.
    
    # Basic usage:
    pipeline = DiffusionPipeline(height=64, width=64, model=model,
                                 text_embedding_function=embed_fn, embedding_dim=384,
                                 num_timesteps=1000, noise_schedule=betas)
    pipeline.generate_image("A scenic mountain landscape", "output.png")
    # With classifier-free guidance:
    pipeline.generate_image("A scenic mountain landscape", "output_cfg.png", cfg=True, c=7)
    """

    def __init__(
        self,
        height: int,
        width: int,
        model,
        text_embedding_function,
        embedding_dim: int,
        num_timesteps: int,
        noise_schedule: jnp.ndarray,
    ):
        # load the trained model
        self.model = model
        self.embed_fn = text_embedding_function

        # diffusion hyper-parameters
        self.num_timesteps = num_timesteps
        self.betas = noise_schedule                 # shape = (T,)
        self.alphas = 1.0 - self.betas              # shape = (T,)
        self.alpha_cumprods = jnp.cumprod(self.alphas)  # shape = (T,)

        # image dimensions
        self.height = height
        self.width = width

        # cache params for unconditional generation
        self.zero_embedding = jnp.zeros((1, 1, embedding_dim))
        self.zero_mask = jnp.ones(shape=(1, 1))

    def generate_image(self, text_prompt: str, output_path: str, cfg: bool = False, c: int = 7):
        """
        Generates an image conditioned on `text_prompt` and saves it to `output_path`. Behavior:
          1. Encodes the provided prompt with `text_embedding_function`.
          2. Initializes a latent image from standard Gaussian noise (dtype=jnp.float32).
          3. Runs a deterministic reverse-diffusion loop from timestep T down to 1:
             - At each step, the model predicts noise for the current noisy image.
             - The pipeline computes a closed-form estimate of x0 (the clean image) using the
               formula x0_hat = (x_t - sqrt(1 - alpha_cumprod_t) * predicted_noise) / sqrt(alpha_cumprod_t).
             - The next latent is computed deterministically from predicted x0 and predicted noise
               (no additional stochastic noise is added during sampling).
             - Intermediate images are clipped to [-1, 1].
          4. After the loop, the final image is mapped from [-1, 1] to [0, 1] and saved with `save_image`.
          5. If `cfg` is True, an additional unconditional sampling pass is performed using `zero_embedding`
             and `zero_mask`, producing `result_unc`. The final image written to disk is computed as:
             final_image = result * c + result_unc, where `c` is a scalar blend weight (default 7).
          6. The method prints the output path and returns None.
        Notes:
          - The model is called with a timestep vector `t_vec = jnp.array([step])` for each step.
          - The implementation uses a deterministic sampler (no Langevin or stochastic term during updates).
          - RNG seeding in the current implementation uses random.PRNGKey(0) combined with a sampled seed;
            callers should be aware of this if determinism across runs is required.
          - `save_image(img_path, img)` is used to persist results; this function is expected to accept
            the image as a jnp.ndarray in [0,1] range.
        """

        # encode prompt
        context, att_mask = self.embed_fn([text_prompt])  # (1, N, C)

        # start from pure Gaussian noise
        seed = random.randint(random.PRNGKey(0), (), 1, 1_000_000)
        key = random.PRNGKey(int(seed))
        image = random.normal(key, (1, self.height, self.width, 3), dtype=jnp.float32)

        # reverse diffusion
        for step in tqdm(range(self.num_timesteps, 0, -1)):
            t_vec = jnp.array([step])
            predicted_noise = self.model(image, t_vec, context, att_mask)

            # beta_t = float(self.betas[step - 1])
            # alpha_t = float(self.alphas[step - 1])
            alpha_cumprod_t = float(self.alpha_cumprods[step - 1])

            sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod = jnp.sqrt(1.0 - alpha_cumprod_t)

            # predict x0
            predicted_x0 = (image - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

            if step > 1:
                alpha_cumprod_prev = float(self.alpha_cumprods[step - 2])
                sqrt_alpha_cumprod_prev = jnp.sqrt(alpha_cumprod_prev)
                sqrt_one_minus_alpha_cumprod_prev = jnp.sqrt(1.0 - alpha_cumprod_prev)

                # deterministic update (no extra noise)
                image = (sqrt_alpha_cumprod_prev * predicted_x0
                         + sqrt_one_minus_alpha_cumprod_prev * predicted_noise)
            else:
                # final step: take x0
                image = predicted_x0


            image = jnp.clip(image, -1.0, 1.0)

        # 4) post-process and save
        result = jnp.squeeze(image, axis=0)
        result = (result + 1.0) * 0.5  # map from [-1,1] to [0,1]

        if not cfg:
            save_image(img_path=output_path, img=result)
            print(f"Image saved to {output_path}")
            return
        
        ### Unconditional generation

        # 2) start from pure Gaussian noise
        image_unc = random.normal(key, (1, self.height, self.width, 3), dtype=jnp.float32)

        # 3) reverse diffusion
        for step in tqdm(range(self.num_timesteps, 0, -1)):
            t_vec = jnp.array([step])
            predicted_noise = self.model(image_unc, t_vec, self.zero_embedding, self.zero_mask)

            # beta_t = float(self.betas[step - 1])
            # alpha_t = float(self.alphas[step - 1])
            alpha_cumprod_t = float(self.alpha_cumprods[step - 1])

            sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod = jnp.sqrt(1.0 - alpha_cumprod_t)

            # predict x0
            predicted_x0 = (image_unc - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

            if step > 1:
                alpha_cumprod_prev = float(self.alpha_cumprods[step - 2])
                sqrt_alpha_cumprod_prev = jnp.sqrt(alpha_cumprod_prev)
                sqrt_one_minus_alpha_cumprod_prev = jnp.sqrt(1.0 - alpha_cumprod_prev)

                # deterministic update (no extra noise)
                image_unc = (sqrt_alpha_cumprod_prev * predicted_x0
                         + sqrt_one_minus_alpha_cumprod_prev * predicted_noise)
            else:
                # final step: take x0
                image_unc = predicted_x0


            image_unc = jnp.clip(image_unc, -1.0, 1.0)

        # 4) post-process and save
        result_unc = jnp.squeeze(image_unc, axis=0)
        result_unc = (result_unc + 1.0) * 0.5  # map from [-1,1] to [0,1]

        final_image = result * c + result_unc

        save_image(img_path=output_path, img=final_image)
        print(f"Image saved to {output_path}")

    def __call__(self, text_prompt: str, output_path: str):
        self.generate_image(text_prompt, output_path)
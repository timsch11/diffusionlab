import jax.numpy as jnp
from jax import random, jit, Array
from util import save_image, postprocess

from tqdm import tqdm


__all__ = [
    "DiffusionPipeline"
]


"""Jittable step functions for image generation with or without cfg"""

@jit
def _image_gen_cfg_step(
    model, 
    image: Array, 
    step: int, 
    c_stack: Array, 
    mask_stack: Array, 
    sqrt_alpha_cumprods: Array, 
    sqrt_one_minus_alpha_cumprods: Array, 
    s: int
):
    
    img_stack = jnp.concatenate([image, image])
    t_stack = jnp.array([step, step])

    unc_cond_prediction = model(img_stack, t_stack, c_stack, mask_stack)
    predicted_noise = unc_cond_prediction[1] + s * (unc_cond_prediction[0] - unc_cond_prediction[1])


    sqrt_alpha_cumprod = sqrt_alpha_cumprods[step - 1]
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprods[step - 1]

    # predict x0
    predicted_x0 = (image - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

    sqrt_alpha_cumprod_prev = sqrt_alpha_cumprods[step - 2]
    sqrt_one_minus_alpha_cumprod_prev = sqrt_one_minus_alpha_cumprods[step - 2]

    # deterministic update (no extra noise)
    image = (sqrt_alpha_cumprod_prev * predicted_x0
                     + sqrt_one_minus_alpha_cumprod_prev * predicted_noise)

    return jnp.clip(image, -1.0, 1.0)


@jit
def _image_gen_cfg_final_step(
    model, 
    image: Array, 
    step: int, 
    c_stack: Array, 
    mask_stack: Array, 
    sqrt_alpha_cumprods: Array, 
    sqrt_one_minus_alpha_cumprods: Array, 
    s: int
):

    img_stack = jnp.concatenate([image, image])
    t_stack = jnp.array([step, step])

    unc_cond_prediction = model(img_stack, t_stack, c_stack, mask_stack)
    predicted_noise = unc_cond_prediction[1] + s * (unc_cond_prediction[0] - unc_cond_prediction[1])

    sqrt_alpha_cumprod = sqrt_alpha_cumprods[step - 1]
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprods[step - 1]

    # predict x0
    predicted_x0 = (image - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

    return jnp.clip(predicted_x0, -1.0, 1.0)


@jit
def _image_gen_step(
    model, 
    image: Array, 
    step: int, 
    context: Array, 
    att_mask: Array, 
    sqrt_alpha_cumprods: Array, 
    sqrt_one_minus_alpha_cumprods: Array
):
    
    t_vec = jnp.array([step])
    
    predicted_noise = model(image, t_vec, context, att_mask)

    sqrt_alpha_cumprod = sqrt_alpha_cumprods[step - 1]
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprods[step - 1]

    # predict x0
    predicted_x0 = (image - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

    sqrt_alpha_cumprod_prev = sqrt_alpha_cumprods[step - 2]
    sqrt_one_minus_alpha_cumprod_prev = sqrt_one_minus_alpha_cumprods[step - 2]

    # deterministic update (no extra noise)
    image = (sqrt_alpha_cumprod_prev * predicted_x0
                    + sqrt_one_minus_alpha_cumprod_prev * predicted_noise)

    return jnp.clip(image, -1.0, 1.0)


@jit
def _image_gen_final_step(
    model, 
    image: Array, 
    step: int, 
    context: Array, 
    att_mask: Array, 
    sqrt_alpha_cumprods: Array, 
    sqrt_one_minus_alpha_cumprods: Array
):
    
    t_vec = jnp.array([step])
    
    predicted_noise = model(image, t_vec, context, att_mask)

    sqrt_alpha_cumprod = sqrt_alpha_cumprods[step - 1]
    sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprods[step - 1]

    # predict x0
    predicted_x0 = (image - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod

    return jnp.clip(predicted_x0, -1.0, 1.0)


class DiffusionPipeline:

    """
    A minimal diffusion sampling pipeline that generates images from text prompts using a provided model
    and text embedding function. The pipeline performs deterministic reverse diffusion (no added noise
    during sampling) from Gaussian noise to an image, supports optional classifier-free guidance (CFG),
    and writes the resulting image to disk.
    Parametersresult
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
    storage_file : str
        Bin file where dataset mean and std is stored

    
    # Basic usage:
    pipeline = DiffusionPipeline(height=64, width=64, model=model,
                                 text_embedding_function=embed_fn, embedding_dim=384,
                                 num_timesteps=1000, noise_schedule=betas)
    pipeline.generate_image("A scenic mountain landscape", "output.png")
    # Without classifier-free guidance:
    pipeline.generate_image("A scenic mountain landscape", "output_cfg.png", cfg=False)
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
        storage_file: str
    ):
        # load the trained model
        self.model = model
        self.embed_fn = text_embedding_function

        # diffusion hyper-parameters
        self.num_timesteps = num_timesteps
        self.betas = noise_schedule                 # shape = (T,)

        # cache alpha params
        self.alphas = 1.0 - self.betas              # shape = (T,)
        self.alpha_cumprods = jnp.cumprod(self.alphas)  # shape = (T,)
        self.sqrt_alpha_cumprod = jnp.sqrt(self.alpha_cumprods)
        self.sqrt_one_minus_alpha_cumprod = jnp.sqrt(1.0 - self.alpha_cumprods)

        # image dimensions
        self.height = height
        self.width = width

        # load mean and std of dataset
        stats = jnp.load(storage_file)
        self.dataset_mean = jnp.asarray(stats["mean"], dtype=jnp.float32)
        self.dataset_std  = jnp.asarray(stats["std"],  dtype=jnp.float32)

        # cache params for unconditional generation
        self.zero_embedding = jnp.zeros((1, 1, embedding_dim))
        self.zero_mask = jnp.ones(shape=(1, 1))

    def generate_images(
        self, 
        *prompts, 
        target_directory: str = "", 
        cfg: bool = True, 
        s: int = 8
    ):
        
        # encode prompt
        context, att_mask = self.embed_fn(prompts)  # (B, N, C)

        b = len(prompts)

        # start from pure Gaussian noise
        seed = random.randint(random.PRNGKey(0), (), 1, 1_000_000)
        key = random.PRNGKey(int(seed))
        images = random.normal(key, (b, self.height, self.width, 3), dtype=jnp.float32)

        # reverse diffusion
        # for cfg generation...
        if cfg:
            # for each img individually (no batching here -> would require microbatch support of the diffusion model)
            for i in range(b):

                # compute image, context and masks for cfg gen
                img = jnp.expand_dims(images[i], axis=0)

                c_i = jnp.expand_dims(context[i], axis=0)
                c_stack = jnp.concatenate([c_i, jnp.zeros(shape=c_i.shape)])

                msk_i = jnp.expand_dims(att_mask[i], axis=0)
                mask_stack = jnp.concatenate([msk_i, jnp.zeros(shape=msk_i.shape)])

                # denoising
                for step in tqdm(range(self.num_timesteps, 1, -1)):
                    img = _image_gen_cfg_step(
                        self.model, 
                        img, 
                        step, 
                        c_stack, 
                        mask_stack, 
                        self.sqrt_alpha_cumprod,
                        self.sqrt_one_minus_alpha_cumprod,
                        s
                    )

                # last step
                img = _image_gen_cfg_final_step(
                    self.model, 
                    img, 
                    1,
                    c_stack, 
                    mask_stack, 
                    self.sqrt_alpha_cumprod,
                    self.sqrt_one_minus_alpha_cumprod,
                    s
                )

                # remove 'batch' dim
                img = jnp.squeeze(img, axis=0)

                # postprocessing
                result = postprocess(img, self.dataset_mean, self.dataset_std)

                # saving
                output_path = f"{target_directory}{prompts[i].replace(" ", "_")}.jpeg"
                save_image(img_path=output_path, img=result)
                print(f"Image saved to {output_path}")

        else:
            # for each img individually (no batching here -> would require microbatch support of the diffusion model)
            for i in range(b):

                # compute image, context and masks for cfg gen
                img = jnp.expand_dims(images[i], axis=0)
                c_i = jnp.expand_dims(context[i], axis=0)
                msk_i = jnp.expand_dims(att_mask[i], axis=0)

                # denoising
                for step in tqdm(range(self.num_timesteps, 1, -1)):
                    img = _image_gen_step(
                        self.model, 
                        img, 
                        step, 
                        c_i, 
                        msk_i, 
                        self.sqrt_alpha_cumprod,
                        self.sqrt_one_minus_alpha_cumprod
                    )

                # last step
                img = _image_gen_final_step(
                    self.model, 
                    img, 
                    1, 
                    c_i, 
                    msk_i, 
                    self.sqrt_alpha_cumprod,
                    self.sqrt_one_minus_alpha_cumprod             
                )

                # remove 'batch' dim
                img = jnp.squeeze(img, axis=0)

                # invert dataset standardization model was trained on
                result = postprocess(img, self.dataset_mean, self.dataset_std)

                # saving
                output_path = f"{target_directory}{prompts[i].replace(" ", "_")}.jpeg"
                save_image(img_path=output_path, img=result)
                print(f"Image saved to {output_path}")


    def generate_image(
        self,
        text_prompt: str, 
        output_path: str, 
        cfg: bool = True, 
        s: int = 8
    ):

        # encode prompt
        context, att_mask = self.embed_fn([text_prompt])  # (1, N, C)

        # start from pure Gaussian noise
        seed = random.randint(random.PRNGKey(0), (), 1, 1_000_000)
        key = random.PRNGKey(int(seed))
        image = random.normal(key, (1, self.height, self.width, 3), dtype=jnp.float32)

        if cfg:
            c_stack = jnp.concatenate([context, jnp.zeros(shape=context.shape)])
            mask_stack = jnp.concatenate([att_mask, jnp.zeros(shape=att_mask.shape)])

        # reverse diffusion
        if cfg:
            for step in tqdm(range(self.num_timesteps, 1, -1)):
                image = _image_gen_cfg_step(
                    self.model, 
                    image, 
                    step, 
                    c_stack, 
                    mask_stack, 
                    self.sqrt_alpha_cumprod,
                    self.sqrt_one_minus_alpha_cumprod,
                    s
                )

            # last step
            image = _image_gen_cfg_final_step(
                self.model, 
                image, 
                1, 
                c_stack, 
                mask_stack, 
                self.sqrt_alpha_cumprod,
                self.sqrt_one_minus_alpha_cumprod, 
                s
            )

        else:
            for step in tqdm(range(self.num_timesteps, 1, -1)):
                image = _image_gen_step(
                    self.model, 
                    image, 
                    step, 
                    context, 
                    att_mask, 
                    self.sqrt_alpha_cumprod,
                    self.sqrt_one_minus_alpha_cumprod
                )

            # last step
            image = _image_gen_final_step(
                self.model, 
                image, 
                1, 
                context, 
                att_mask, 
                self.sqrt_alpha_cumprod,
                self.sqrt_one_minus_alpha_cumprod
            )

        # remove batch dim
        result = jnp.squeeze(image, axis=0)

        # invert dataset standardization model was trained on
        result = postprocess(result, self.dataset_mean, self.dataset_std)

        # saving
        save_image(img_path=output_path, img=result)
        print(f"Image saved to {output_path}")
        

    def __call__(self, text_prompt: str, output_path: str, cfg: bool = True, s: int = 8):

        """
        Calls 'generate_image'. Generates an image using the prompt as a condition and stores it at the specified location. 
        Classifier free guidance is enabled by default. 
        Change cfg flag or s param to disable or change the cfg weighting.

        """

        self.generate_image(text_prompt, output_path, cfg, s)

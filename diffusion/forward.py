from random import randint as pyrandom

import jax.numpy as jnp
from jax import Array, random
from flax.nnx import jit


def apply_t_noise_steps(img: Array, t: int, betas: Array, dtype: jnp.dtype = jnp.float32, PRNGKey: random.key = None) -> Array:
    """
    Applies t steps of gaussian noise to the given image

    Parameters:
        img (jax.Array): Normalized array representing the image to apply noise to
        t (int): Amount of noising steps to be applied
        betas (jax.Array): Array of beta value (Important: length of betas must be equal to t)

    Returns:
        Array representation of image after t steps of noise applies

    """

    # calculate cumulative product of (1 - beta_i)
    alphas = 1.0 - betas
    alpha = jnp.cumprod(alphas)[t-1]

    # calculate sqrt(alpha) and sqrt(1-alpha)
    sqrt_alpha = jnp.sqrt(alpha)
    sqrt_ialpha = jnp.sqrt(1 - alpha)

    # generate random key if neccessary
    if PRNGKey is None:
        PRNGKey = random.key(pyrandom(1, 100000000))

    # generate noise from N~(0,1)
    noise = random.normal(key=PRNGKey, shape=img.shape, dtype=dtype)

    # apply noise to img
    return sqrt_alpha * img + sqrt_ialpha * noise


def apply_noise_step(img: Array, beta: float, dtype: jnp.dtype = jnp.float32, PRNGKey: random.key = None) -> Array:
    """
    Applies a single step of gaussian noise to the given image

    Parameters:
        img (jax.Array): Normalized array representing the image to apply noise to
        betas(float): beta value 

    Returns:
        Array representation of image after a single step of gaussian noise

    """

    # calculate cumulative product of (1 - beta_i)
    sqrt_alpha = (1 - beta) ** (1/2)
    sqrt_beta = beta ** (1/2)

    # generate random key if neccessary
    if PRNGKey is None:
        PRNGKey = random.key(pyrandom(1, 100000000))

    # generate noise from N~(0,1)
    noise = random.normal(key=PRNGKey, shape=img.shape, dtype=dtype)

    # apply noise to img
    return sqrt_alpha * img + sqrt_beta * noise


def noisify(img: Array, t: int, betas: Array, dtype: jnp.dtype = jnp.float32, key: random.key = None):
    """
    Returns x_t and the noise eps such that
      x_t = sqrt(alpha_t) * x0 + sqrt(1 - alpha_t) * eps
    """

    alphas = 1.0 - betas
    alpha_cumprod = jnp.cumprod(alphas)[t]
    sqrt_alpha = jnp.sqrt(alpha_cumprod)
    sqrt_ialpha = jnp.sqrt(1.0 - alpha_cumprod)

    ε = random.normal(key=key, shape=img.shape, dtype=dtype)
    x_t = sqrt_alpha * img + sqrt_ialpha * ε
    return x_t, ε


# Explicitly compile functions, wrapper gave no speedup for static params
apply_t_noise_steps = jit(apply_t_noise_steps, static_argnames=("dtype", ))
apply_noise_step = jit(apply_noise_step, static_argnames=("dtype", ))
noisify = jit(noisify, static_argnames=("dtype",))
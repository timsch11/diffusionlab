from random import randint as pyrandom

import jax.numpy as jnp
from jax import Array, random, jit


def apply_noise(img: Array, t: int, betas: Array, PRNGKey: random.key = None) -> Array:
    """
    Applies t steps of gaussian noise to the given image

    Parameters:
        img (jax.Array): Normalized array representing the image to apply noise to
        t (int): Amount of noising steps to be applied
        betas (jax.Array): Array of beta value (Important: length of betas must be equal to t)

    Returns:
        Array representation of image after t steps of noise applies

    """

    # shape checking
    if betas.shape[0] != t:
        raise ValueError("fIncompatible betas shape for timestamp {t}")

    # calculate cumulative product of (1 - beta_i)
    alphas = 1.0 - betas
    alpha = jnp.cumprod(alphas)[-1]

    # calculate sqrt(alpha) and sqrt(1-alpha)
    sqrt_alpha = jnp.sqrt(alpha)
    sqrt_ialpha = jnp.sqrt(1 - alpha)

    # generate random key if neccessary
    if PRNGKey is None:
        PRNGKey = random.key(pyrandom(1, 100000000))

    # generate noise from N~(0,1)
    noise = random.normal(key=PRNGKey, shape=img.shape)

    # apply noise to img
    return sqrt_alpha * img + sqrt_ialpha * noise
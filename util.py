from PIL import Image
import numpy as np
import jax.numpy as jnp
from jax import Array
import orbax.checkpoint as orbax
from flax import nnx

import os


def load_image(img_path: str, dtype: jnp.dtype = jnp.bfloat16, normalize: bool = True) -> Array:
    """Loads image from path and loads it into a jax array of <dtype>, optionally normalizes array"""
    
    # Load image with pillow
    pil_img = Image.open(img_path).convert("RGB")

    # Convert to NumPy array (shape: H × W × C)
    np_img = np.array(pil_img)

    # Convert into a JAX array
    jax_img = jnp.asarray(np_img).astype(dtype)

    # Normalize (if required)
    if normalize:
        return (jax_img / 127.5) - 1
    
    return jax_img


def rescale_image(img_path: str, target_width: int, target_height: int, dtype: jnp.dtype = jnp.bfloat16, normalize: bool = True) -> Array:
    """Rescales an image to the specified resolution and returns as a JAX array.
    
    Args:
        img_path: Path to the input image
        target_width: Target width in pixels
        target_height: Target height in pixels
        dtype: JAX data type for the output array
        normalize: Whether to normalize pixel values to [-1, 1]
    
    Returns:
        JAX array with shape (target_height, target_width, 3)
    """

    # Load image as PIL Image first (before converting to JAX)
    pil_img = Image.open(img_path).convert("RGB")
    
    # Resize the PIL image
    resized_pil = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Convert to NumPy then JAX array
    np_img = np.array(resized_pil)
    jax_img = jnp.asarray(np_img).astype(dtype)
    
    # Normalize if required
    if normalize:
        return (jax_img / 127.5) - 1
    
    return jax_img


def save_image(img_path: str, img: Array) -> None:
    img_uint8 = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
    # Convert JAX array to NumPy array before passing to PIL
    img_np = np.asarray(img_uint8)
    mode = 'RGB' if img_np.ndim == 3 and img_np.shape[2] == 3 else None
    pil_img = Image.fromarray(img_np, mode)
    pil_img.save(img_path, format="JPEG", quality=95)


def save_model(model, path: str):
    # Retrive state (params)
    state = nnx.state(model)

    # Convert to absolute path
    abs_path = os.path.abspath(path)

    # Save the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(abs_path, state, force=True)


def load_model(model_without_params, path: str):
    state = nnx.state(model_without_params)

    # Convert to absolute path
    abs_path = os.path.abspath(path)

    # Load the parameters
    checkpointer = orbax.PyTreeCheckpointer()
    state = checkpointer.restore(abs_path, item=state)

    # update the model with the loaded state
    nnx.update(model_without_params, state)

    return model_without_params
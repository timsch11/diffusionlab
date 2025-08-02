from PIL import Image
import numpy as np
import jax.numpy as jnp
from jax import Array


def load_image(img_path: str, dtype: jnp.dtype = jnp.bfloat16, normalize: bool = False) -> Array:
    """Loads image from path and loads it into a jax array of <dtype>, optionally normalizes array"""
    
    # Load image with pillow
    pil_img = Image.open(img_path).convert("RGB")

    # Convert to NumPy array (shape: H × W × C)
    np_img = np.array(pil_img)

    # Convert into a JAX array
    jax_img = jnp.asarray(np_img).astype(dtype)

    # Normalize (if required)
    if normalize:
        return jax_img / 255.0
    
    return jax_img


def rescale_image(img_path: str, target_width: int, target_height: int, dtype: jnp.dtype = jnp.bfloat16, normalize: bool = False) -> Array:
    """Rescales an image to the specified resolution and returns as a JAX array.
    
    Args:
        img_path: Path to the input image
        target_width: Target width in pixels
        target_height: Target height in pixels
        dtype: JAX data type for the output array
        normalize: Whether to normalize pixel values to [0, 1]
    
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
        return jax_img / 255.0
    
    return jax_img


def save_image(img_path: str, img: Array) -> None:
    img_uint8 = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
    # Convert JAX array to NumPy array before passing to PIL
    img_np = np.asarray(img_uint8)
    mode = 'RGB' if img_np.ndim == 3 and img_np.shape[2] == 3 else None
    pil_img = Image.fromarray(img_np, mode)
    pil_img.save(img_path, format="JPEG", quality=95)
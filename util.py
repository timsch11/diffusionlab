from PIL import Image
import numpy as np
import jax.numpy as jnp
from jax import Array


def load_image(img_path: str, dtype: jnp.dtype = jnp.float32, normalize: bool = False) -> Array:
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


def save_image(img_path: str, img: Array) -> None:
    img_uint8 = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
    # Convert JAX array to NumPy array before passing to PIL
    img_np = np.asarray(img_uint8)
    mode = 'RGB' if img_np.ndim == 3 and img_np.shape[2] == 3 else None
    pil_img = Image.fromarray(img_np, mode)
    pil_img.save(img_path, format="JPEG", quality=95)
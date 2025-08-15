import jax.numpy as jnp


def get_2d_sinusoidal_positional_encoding(H: int, W: int, C: int) -> jnp.ndarray:
    """
    Generate a fixed 2D sinusoidal positional encoding in flattened form.

    Args:
        H (int): Image height
        W (int): Image width
        C (int): Embedding dimension (must be divisible by 4)

    Returns:
        pos_enc_flat: [H*W, C] JAX array
    """
    assert C % 4 == 0, "Embedding dimension must be divisible by 4 for 2D encoding."

    # Position indices
    y_pos = jnp.arange(H)[:, None]  # [H,1]
    x_pos = jnp.arange(W)[:, None]  # [W,1]

    # Frequencies for sine/cosine
    div_term_y = jnp.exp(jnp.arange(0, C // 2, 2) * -(jnp.log(10000.0) / (C // 2)))
    div_term_x = jnp.exp(jnp.arange(0, C // 2, 2) * -(jnp.log(10000.0) / (C // 2)))

    # Encode Y positions
    pos_y = jnp.zeros((H, W, C // 2))
    angles_y = y_pos * div_term_y  # [H, C//4]
    pos_y = pos_y.at[:, :, 0::2].set(jnp.sin(angles_y)[:, None, :])  # [H,1,C//4] -> broadcast over W
    pos_y = pos_y.at[:, :, 1::2].set(jnp.cos(angles_y)[:, None, :])  # [H,1,C//4] -> broadcast over W

    # Encode X positions
    pos_x = jnp.zeros((H, W, C // 2))
    angles_x = x_pos * div_term_x  # [W, C//4]
    pos_x = pos_x.at[:, :, 0::2].set(jnp.sin(angles_x)[None, :, :])  # [1,W,C//4] -> broadcast over H
    pos_x = pos_x.at[:, :, 1::2].set(jnp.cos(angles_x)[None, :, :])  # [1,W,C//4] -> broadcast over H

    # Concatenate Y and X encodings
    pos_enc = jnp.concatenate([pos_y, pos_x], axis=-1)  # [H, W, C]

    # Flatten spatial dims to [H*W, C]
    pos_enc_flat = pos_enc.reshape(H * W, C)
    return pos_enc_flat
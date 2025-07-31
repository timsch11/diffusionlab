import jax.numpy as jnp


arr = jnp.array([0.1 for i in range(10)])

print(jnp.cumprod(arr))
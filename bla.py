import jax
import time
import random
import jax.numpy as jnp
from tqdm import tqdm


val = 1
for i in tqdm(range(100), desc=f"dawdwd val: {val}"):
    val += 2*3
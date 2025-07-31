import jax
import time
import random


@jax.jit
def func(lim: int) -> list:
    primes = [1 for _ in range(lim+1)]
    primes[0] = 0
    primes[1] = 0
    for i in range(2, lim+1):
        if primes[i] == 1:
            for j in range(2, ((lim+1) // i)):
                primes[j*i] = 0

    return primes


@jax.jit
def func2(r: int, key: jax.Array):
    
    arr = jax.random.uniform(key=key, shape=(100000000, ))

    arr2 = arr ** 3

    return arr2


LIMIT = 10000000

# Create a random key for JAX
key = jax.random.PRNGKey(42)

func2(1, key)

t = time.time()
func2(1, key)
print(time.time() - t)
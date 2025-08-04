from dataloader import Dataloader
from schedule import cosine_beta_schedule
import jax.numpy as jnp
import time
import random


t = time.time()

dl = Dataloader("emojiimage-dataset/image/Google", "emojiimage-dataset/full_emoji.csv", 64, 64, 200, cosine_beta_schedule(200), 4, jnp.float32)

print("Init time: ", time.time() - t)

t1 = time.time()

for x, t, x_embedd, y in dl:
    print(x_embedd)
    break


print("Batch time (1): ", time.time() - t1)
t2 = time.time()

for x, t, x_embedd, y in dl:
    print(x_embedd)
    break

print("Batch time (2): ", time.time() - t2)
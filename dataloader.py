import jax.numpy as jnp
from jax import Array, devices, device_put
import pandas as pd
from prompt_embedding import embedd_prompts_batched
from diffusion.forward import apply_t_noise_steps, apply_noise_step
from util import rescale_image
import random


CPU = devices('cpu')[0]
GPU = devices('cuda')[0]


class Dataloader:
    def __init__(self, data_dir: str, csv_file_path: str, target_height: int, target_width: int, timesteps: int, schedule: Array, batch_size: int, dtype: jnp.dtype):
        df = pd.read_csv(csv_file_path)

        min_index = df['#'].min()
        max_index = df['#'].max()

        imgs = list()
        for i in range(min_index, max_index+1):
            img_path = f"{data_dir}/{i}.png"

            imgs.append(rescale_image(img_path=img_path, target_height=target_height, target_width=target_width, dtype=dtype, normalize=True))

        imgs = jnp.stack(imgs)
        self.imgs = device_put(imgs, device=CPU)

        if schedule.shape[0] != timesteps:
            raise ValueError("Incompatible schedule for given timesteps")
        
        self.timesteps = timesteps
        self.schedule = schedule

        self.dtype = dtype

        self.raw_data: list[Array, Array] = list()  # [[img_arr, text_embedding_arr]]

        prompts = df['name'].to_list()
        embedded_prompts = embedd_prompts_batched(prompts, chunk_size=100).astype(dtype=dtype)
        self.embedding = device_put(embedded_prompts, device=CPU)

        # precompute measures for batch sampling
        self.num_items = self.embedding.shape[0]
        self.batch_size = batch_size

        self.batch_initalized = False
        

    def load_batches(self):
        indicies = [_ for _ in range(self.num_items)]
        random.shuffle(indicies)

        self.epoch_x = list()
        self.epoch_y = list()
        self.epoch_x_embedd = list()
        self.epoch_t = list()

        batch = 0
        upper_bound = self.batch_size * (self.num_items // self.batch_size)
        while batch < upper_bound:
            
            batch_x = list()
            batch_y = list()
            batch_x_embedd = list()
            batch_t = list()

            for j in range(batch, batch + self.batch_size):

                i = indicies[j]

                t = j % self.timesteps  # i and indecies[i] are uncorrelated since imgs is shuffled ahead of each batch preparation

                if t == 0:
                    label = self.imgs[i]
                
                else:
                    label = apply_t_noise_steps(self.imgs[i], t, self.schedule[:t], dtype=self.dtype)

                label = device_put(label, device=GPU)
                sample = apply_noise_step(label, self.schedule[t], dtype=self.dtype)  # sample is on GPU -> operation is performed on GPU -> label is on GPU
                embedding = device_put(self.embedding[i], device=GPU)

                batch_x.append(sample)
                batch_y.append(label)
                batch_x_embedd.append(embedding)
                batch_t.append(t)

            # stack to batch tensor and append tensor to list
            self.epoch_x.append(jnp.stack(batch_x))
            self.epoch_y.append(jnp.stack(batch_y))
            self.epoch_x_embedd.append(jnp.stack(batch_x_embedd))
            self.epoch_t.append(jnp.stack(batch_t))

            batch += self.batch_size

        # last potentially incomplete batch 
        batch_x = list()
        batch_y = list()
        batch_x_embedd = list()
        batch_t = list()
        
        for j in range(upper_bound, self.num_items):
            i = indicies[j]

            t = j % self.timesteps  # i and indecies[i] are uncorrelated since imgs is shuffled ahead of each batch preparation
            if t == 0:
                label = self.imgs[i]
                
            else:
                label = apply_t_noise_steps(self.imgs[i], t, self.schedule[:t], dtype=self.dtype)

            label = device_put(label, device=GPU)
            sample = apply_noise_step(label, self.schedule[t], dtype=self.dtype)  # sample is on GPU -> operation is performed on GPU -> label is on GPU
            embedding = device_put(self.embedding[i], device=GPU)

            batch_x.append(sample)
            batch_y.append(label)
            batch_x_embedd.append(embedding)
            batch_t.append(t)

        if len(batch_x) > 0:
            # stack to batch tensor and append tensor to list
            self.epoch_x.append(jnp.stack(batch_x))
            self.epoch_y.append(jnp.stack(batch_y))
            self.epoch_x_embedd.append(jnp.stack(batch_x_embedd))
            self.epoch_t.append(jnp.stack(batch_t))

        self.batch_initalized = True

        print("length x", len(self.epoch_x))
        print("length y", len(self.epoch_y))
        print("length embedd", len(self.epoch_x_embedd))
        print("length t", len(self.epoch_t))


    def __iter__(self):
        if not self.batch_initalized:
            self.load_batches()

        self.batch_initalized = False
        self.i = -1
        self.num_batches = len(self.epoch_x)

        return self

    def __next__(self):
        self.i += 1
        if self.i == self.num_batches:
            raise StopIteration
        
        return self.epoch_x[self.i], self.epoch_t[self.i], self.epoch_x_embedd[self.i], self.epoch_y[self.i]
    
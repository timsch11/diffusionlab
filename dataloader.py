import jax.numpy as jnp
from jax import Array, devices, device_put
import pandas as pd
from prompt_embedding import embedd_prompts_batched
from util import rescale_image


CPU = devices('cpu')[0]

class Dataloader:
    def __init__(self, data_dir: str, csv_file_path: str, target_height: int, target_width: int, batch_size: int, dtype: jnp.dtype):
        df = pd.read_csv(csv_file_path)

        min_index = df['#'].min()
        max_index = df['#'].max()

        print(df["name"])

        imgs = list()
        for i in range(min_index, max_index+1):
            img_path = f"{data_dir}/{i}.png"

            imgs.append(rescale_image(img_path=img_path, target_height=target_height, target_width=target_width, dtype=dtype, normalize=True))

        imgs = jnp.stack(imgs)
        imgs = device_put(imgs, device=CPU)

        self.raw_data: list[Array, Array] = list()  # [[img_arr, text_embedding_arr]]

        prompts = df['name'].to_list()
        embedded_prompts = embedd_prompts_batched(prompts, chunk_size=100).astype(dtype=dtype)
        self.embedding = device_put(embedded_prompts, device=CPU)

        #for index, row in df.iterrows():
         #   row



if __name__ == '__main__':
    dl = Dataloader("emojiimage-dataset/image/Google", "emojiimage-dataset/full_emoji.csv", 64, 64, 4, jnp.float32)
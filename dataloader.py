# dataloader.py (reworked)
import jax
import jax.numpy as jnp
from jax import random, devices, device_put
import numpy as np
import pandas as pd

from prompt_embedding import embedd_prompts_seq
from diffusion.forward import noisify 
from util import rescale_image


CPU = devices('cpu')[0]
GPU = devices('cuda')[0]


class Dataloader:
    """
    Dataloader for diffusion models that loads and processes image data with corresponding text embeddings.

    This class handles:
    - Loading and normalizing images from a directory
    - Processing text embeddings for conditioning
    - Batching data with deterministic per-epoch shuffling
    - Implementing classifier-free guidance dropout
    - Generating noisy samples at random timesteps for diffusion training

    Parameters
    ----------
    data_dir : str
        Directory containing the image files (named as {index}.png)
    csv_file_path : str
        Path to CSV file containing image metadata with '#' column for indices and 'name' column for text prompts
    target_height : int
        Target height for resizing images
    target_width : int
        Target width for resizing images
    embedding_dim : int
        Dimension of the text embeddings
    embedding_dropout : float
        Probability of dropping text embeddings for cfg training
    timesteps : int
        Number of diffusion timesteps
    schedule : jnp.ndarray
        Noise schedule array for the diffusion process
    batch_size : int
        Number of samples per batch
    dtype : jnp.dtype
        Data type for arrays (e.g., jnp.float32)
    key : jax.Array
        PRNG key for reproducible randomness
    max_index : int, optional
        Maximum index to include from dataset, defaults to -1 (use all available data)

    Methods
    -------
    load_batches()
        Prepares batches for the current epoch with deterministic shuffling
    _epoch_permutation()
        Generates a deterministic permutation of indices for the current epoch

    Yields
    ------
    tuple
        Each iteration yields (noisy_images, timesteps, text_embeddings, attention_masks, noise_targets)
    """

    def __init__(
        self,
        data_dir: str,
        csv_file_path: str,
        target_height: int,
        target_width: int,
        embedding_dim: int,
        embedding_dropout: float,
        timesteps: int,
        schedule: jnp.ndarray,
        batch_size: int,
        dtype: jnp.dtype,
        key: jax.Array,
        max_index: int = -1
    ):
        df = pd.read_csv(csv_file_path)
        min_index = int(df['#'].min())
        max_index = max_index if max_index != -1 else int(df['#'].max())

        # Load & normalize images once on CPU
        imgs = []
        for i in range(min_index, max_index + 1):
            img_path = f"{data_dir}/{i}.png"
            imgs.append(rescale_image(
                img_path=img_path,
                target_height=target_height,
                target_width=target_width,
                dtype=dtype,
                normalize=True
            ))
        imgs = jnp.stack(imgs)
        self.imgs = device_put(imgs, device=CPU)

        if schedule.shape[0] != timesteps:
            raise ValueError("Incompatible schedule for given timesteps")

        self.timesteps = timesteps
        self.schedule = schedule.astype(jnp.float32)  # math stability
        self.dtype = dtype
        self.batch_size = batch_size

        # Text embeddings (per token) + masks
        prompts = df['name'].to_list()
        embedded_prompts, att_masks = embedd_prompts_seq(prompts[:max_index+1])  # returns [B,T,D], [B,T]
        self.embedding = device_put(embedded_prompts.astype(dtype), device=CPU)
        self.att_masks = device_put(att_masks, device=CPU)

        self.num_items = self.embedding.shape[0]
        self.seq_len = int(self.embedding.shape[1])
        self.embedding_dim = int(embedding_dim)

        # Zero context for unconditional passes (shape [T, D])
        zero_vec = jnp.zeros((self.seq_len, self.embedding_dim), dtype=dtype)
        self.zero_vec = device_put(zero_vec, device=GPU)

        # CFG dropout prob
        self.p_drop = float(embedding_dropout)

        # RNG: keep ONE base key; fold in epoch & sample indices
        self.base_key = key
        self.epoch = 0

        # Caches for one epoch
        self.batch_initialized = False

    def _epoch_permutation(self):
        """Deterministic per-epoch permutation using fold_in so it doesn't depend on batch size."""
        key_perm = random.fold_in(self.base_key, self.epoch)
        perm = random.permutation(key_perm, self.num_items)
        return np.array(perm)  # convenient for Python indexing

    def load_batches(self):
        indices = self._epoch_permutation()

        self.epoch_x = []
        self.epoch_y = []
        self.epoch_x_embedd = []
        self.epoch_t = []
        self.epoch_att_msk = []

        # Build batches sequentially; RNG per-sample via fold_in(pos)
        pos = 0
        while pos < self.num_items:
            j_end = min(pos + self.batch_size, self.num_items)
            batch_idx = indices[pos:j_end]

            batch_x, batch_eps = [], []
            batch_emb, batch_t, batch_mask = [], [], []

            for local_j, i in enumerate(batch_idx):
                # Per-sample RNG stream independent of batching
                # Seed by (epoch, global_position)
                k_base = random.fold_in(self.base_key, (self.epoch << 20) + int(pos + local_j))
                k_t = random.fold_in(k_base, 0)
                k_noise = random.fold_in(k_base, 1)
                k_cfd = random.fold_in(k_base, 2)

                # Sample t ∈ [0, T-1] (0-based)
                t = int(random.randint(k_t, shape=(), minval=0, maxval=self.timesteps))

                # Get x_t and ε using forward noising with a key
                x0 = self.imgs[i]  # CPU
                # If your noisify still expects 1-based t and schedule[:t], adapt:
                # t_plus = t + 1
                # x_t, eps = noisify(k_noise, x0, t_plus, self.schedule[:t_plus], dtype=self.dtype)
                x_t, eps = noisify(x0, t, self.schedule, dtype=self.dtype, key=k_noise)  # preferred 0-based API

                # Move to GPU
                x_t = device_put(x_t, device=GPU)
                eps = device_put(eps, device=GPU)

                # Classifier-free guidance dropout (drop=1 → use zero context)
                keep = random.bernoulli(k_cfd, p=1.0 - self.p_drop)
                if bool(keep):
                    emb = device_put(self.embedding[i], device=GPU)
                    msk = device_put(self.att_masks[i], device=GPU)
                else:
                    emb = self.zero_vec
                    # When dropping, you can set mask to zeros or keep the original mask.
                    # Zeros means the cross-attn sees no valid tokens:
                    msk = device_put(jnp.zeros_like(self.att_masks[i]), device=GPU)

                batch_x.append(x_t)
                batch_eps.append(eps)
                batch_emb.append(emb)
                batch_t.append(t+1)
                batch_mask.append(msk)

            # Stack this batch
            self.epoch_x.append(jnp.stack(batch_x))
            self.epoch_y.append(jnp.stack(batch_eps))
            self.epoch_x_embedd.append(jnp.stack(batch_emb))
            self.epoch_t.append(jnp.array(batch_t))
            self.epoch_att_msk.append(jnp.stack(batch_mask))

            pos = j_end

        self.batch_initialized = True
        self.epoch += 1  # advance for next call

    def __iter__(self):
        if not self.batch_initialized:
            self.load_batches()
        self.batch_initialized = False
        self.i = 0
        self.num_batches = len(self.epoch_x)
        return self

    def __next__(self):
        if self.i >= self.num_batches:
            raise StopIteration
        out = (
            self.epoch_x[self.i],
            self.epoch_t[self.i],
            self.epoch_x_embedd[self.i],
            self.epoch_att_msk[self.i],
            self.epoch_y[self.i],
        )
        self.i += 1
        return out

if __name__ == '__main__':
    pass  # dataloader = Dataloader(data_dir="emojiimage-dataset/image/Google", csv_file_path="emojiimage-dataset/full_emoji.csv", target_height=H, target_width=W, embedding_dim = 384, embedding_dropout=0.1, timesteps=T, schedule=SCHEDULE, batch_size=B, dtype=jnp.float32)

    
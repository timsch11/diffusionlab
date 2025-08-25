import jax
import jax.numpy as jnp
from jax import random, devices, device_put
import numpy as np
import pandas as pd

from diffusion.prompt_embedding import embedd_prompts_seq
from diffusion.forward import noisify 
from util import rescale_image, standardize


CPU = devices('cpu')[0]
GPU = devices('cuda')[0]

# Stateless, jittable helper that builds a whole batch given indices and integer seeds.
def _make_batch(imgs, embedding, att_masks, zero_vec, schedule, p_drop, timesteps, dtype, idxs, seeds):
    """
    imgs: [N,...]
    embedding: [N, T, D]
    att_masks: [N, T]
    zero_vec: [T, D]
    idxs: [B] int indices into imgs/embedding
    seeds: [B] deterministic integer seeds (e.g. epoch-based)
    Returns: x_t [B,...], eps [B,...], emb [B,T,D], t_out [B], mask [B,T]
    """
    def sample_fn(idx, seed):
        x0 = imgs[idx]
        key = random.PRNGKey(seed)
        k_t, k_noise, k_cfd = random.split(key, 3)
        t = random.randint(k_t, shape=(), minval=0, maxval=timesteps)  # 0-based
        x_t, eps = noisify(x0, t, schedule, dtype=dtype, key=k_noise)
        keep = random.bernoulli(k_cfd, p=1.0 - p_drop)
        # keep is scalar boolean; broadcasting works to select full token arrays
        emb = jnp.where(keep, embedding[idx], zero_vec)
        msk = jnp.where(keep, att_masks[idx], jnp.zeros_like(att_masks[idx]))
        return x_t, eps, emb, t, msk

    # vmap over batch dimension
    x_t_out, eps_out, emb_out, t_out, msk_out = jax.vmap(sample_fn)(idxs, seeds)
    return x_t_out, eps_out, emb_out, t_out, msk_out


_make_batch = jax.jit(_make_batch, static_argnames=("dtype",))

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
        max_index: int = -1,
        file_storage: str = "dataset_stats.npz"
    ):
        
        df = pd.read_csv(csv_file_path)
        min_index = int(df['#'].min())
        max_index = max_index if max_index != -1 else int(df['#'].max())

        # Load & normalize images once on CPU
        imgs = []

        for i in range(min_index, max_index + 1):
            img_path = f"{data_dir}/{i}.png"
            img = rescale_image(
                img_path=img_path,
                target_height=target_height,
                target_width=target_width,
                dtype=dtype,
            )
            imgs.append(img)

        imgs = jnp.stack(imgs)

        # standardize dataset and save mean and std
        imgs, mean, std = standardize(imgs)

        # cache measures in bin file
        jnp.savez(file_storage, mean=mean, std=std)

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

        # Build batches sequentially; RNG per-sample via deterministic integer seeds
        pos = 0
        while pos < self.num_items:
            j_end = min(pos + self.batch_size, self.num_items)
            batch_idx = indices[pos:j_end]

            # Prepare deterministic integer seeds (one per sample) derived from epoch and global position.
            # Keep the same addressing scheme as before: (epoch << 20) + global_pos
            num_local = len(batch_idx)
            seeds_np = np.array([((self.epoch % 2000) << 20) + int(pos + j) for j in range(num_local)], dtype=np.int32)

            # Convert to jax arrays and call the jitted, stateless batch builder
            idxs_j = jnp.array(batch_idx, dtype=jnp.int32)
            seeds_j = jnp.array(seeds_np, dtype=jnp.int32)

            imgs = device_put(self.imgs, device=GPU)
            embeds = device_put(self.embedding, device=GPU)
            msks = device_put(self.att_masks, device=GPU)

            batch_x_arr, batch_eps_arr, batch_emb_arr, batch_t_arr, batch_mask_arr = _make_batch(
                imgs,
                embeds,
                msks,
                self.zero_vec,
                self.schedule,
                self.p_drop,
                self.timesteps,
                self.dtype,
                idxs_j,
                seeds_j
            )

            # Append produced batch arrays
            self.epoch_x.append(batch_x_arr)
            self.epoch_y.append(batch_eps_arr)
            self.epoch_x_embedd.append(batch_emb_arr)
            self.epoch_t.append(batch_t_arr)
            self.epoch_att_msk.append(batch_mask_arr)

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
    from params import SCHEDULE, RANDOMKEY
    dataloader = Dataloader(data_dir="emojiimage-dataset/image/Google", csv_file_path="emojiimage-dataset/full_emoji.csv", target_height=64, target_width=64, embedding_dim = 384, embedding_dropout=0.1, timesteps=250, schedule=SCHEDULE, batch_size=32, dtype=jnp.float32, key=RANDOMKEY)
    for a in dataloader:
        pass

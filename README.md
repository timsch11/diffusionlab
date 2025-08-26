***Still in development***

A personal portfolio project: a Flax-based implementation of a conditional Denoising Diffusion Probabilistic Model (DDPM / diffusion U-Net) built from scratch to learn how diffusion models work end-to-end.

This repository contains an educational reimplementation of a conditional diffusion image model trained on an emoji dataset (~1.8k images). The goal was to explore building and training a diffusion U‑Net with time embeddings, attention / cross‑attention for text conditioning, and a minimal sampling pipeline using Flax and JAX.

Core ideas
- Build a compact U‑Net style diffusion model in Flax/nnx with ResNet blocks, attention, and timestamp embeddings.
- Condition generation on text using pretrained text encoders (BAAI/bge-small-en) via cross‑attention and support classifier‑free guidance during sampling.
- Train a pixel-wise denoising objective (predict noise / ε) with a cosine beta noise schedule and deterministic sampling for evaluation.

Highlights
- Implemented from scratch using Flax (nnx) and JAX primitives.
- U‑Net architecture split into encoder / bottleneck / decoder with optional self / cross attention at different resolutions.
- Text conditioning via token-level embeddings (cross attention) and support for classifier‑free guidance (CFG) during training and sampling.
- Dataset: emoji images (~1800) from `emojiimage-dataset/` with accompanying CSV metadata `emojiimage-dataset/full_emoji.csv`.
- Checkpointing with Orbax; sample outputs saved to `validation_images/` and model checkpoints under `models_vA0/`.

Quick status
- This is an educational/portfolio project rather than a production training run. Expect the model and scripts to be opinionated and designed for experimentation.

Getting started
1. Prerequisites
   - Linux with CUDA (recommended) and a modern GPU for training/inference. The project uses JAX + CUDA.
   - Python 3.10+ (see `requirements.txt`). Install dependencies in a virtualenv or conda env. Example:
     - pip install -r requirements.txt
   - Make sure JAX is installed for your CUDA version (the `requirements.txt` contains a pinned jax/jaxlib and jax‑cuda packages used during development).

2. Prepare the dataset
   - The repo expects `emojiimage-dataset/` to contain images under `image/<vendor>/<index>.png` and a CSV `emojiimage-dataset/full_emoji.csv` that contains a `#` index column and `name` prompt column.
   - By default `train.py` uses `emojiimage-dataset/image/Google` and `full_emoji.csv` — update paths or `params.MAX_INDEX` to control subset size.

3. Configuration
   - Edit `params.py` to tune H, W, B, T (timesteps), T_dim/T_out timestamp dims, BASE_DIM and `MAX_INDEX` to limit dataset size for faster experiments.
   - GPU memory behavior: `train.py` sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` to let JAX preallocate memory. Adjust as needed.

4. Training (basic)
   - Run: python train.py
   - Checkpoints: saved periodically to `models_vA0/epoch{N}` and final checkpoint to `models_vA0/final`.
   - Sample images are written to `validation_images/` during training.

5. Sampling / Inference
   - Use `test.py` to load the final checkpoint and run a few example prompts. The sampling pipeline uses a deterministic sampler by default (no stochastic term added during reverse steps) and supports simple CFG blending.
   - You can also import `DiffusionPipeline` and call `generate_image(prompt, output_path, cfg=True, c=7)` to enable classifier‑free guidance blending.

Design & architecture notes
- U‑Net backbone: implemented with `nnx.Conv`, `ResNet` residual blocks, nearest‑neighbour upsampling and optional MultiHeadAttention blocks.
- Timestamp embedding: sinusoidal encoding projected to injection dimension via `TimestampNet`.
- Cross-attention: token-level embeddings are passed to attention layers in encoder/decoder blocks where configured.
- Loss: L2 (MSE) between predicted noise and ground‑truth noise. Training objective implemented in `train.py`.
- Noise schedule: cosine schedule implemented in `schedule.py`.

Dataset and conditioning
- The dataset contains emoji images where each image is paired with a text description from the CSV. The dataloader builds tokenized contexts and attention masks (per token) for cross‑attention.
- Classifier‑free guidance: implemented by randomly replacing the context with zeros during training with probability `embedding_dropout` in the dataloader. During sampling, an unconditional pass can be computed and blended.

Tips and caveats
- This project is experimental and tuned for learning rather than production. Expect debug prints, hardcoded paths and a minimal experiment management approach.
- Training from scratch is compute intensive. Use `params.MAX_INDEX` to limit dataset size when experimenting.
- The repository pins specific JAX / Flax / CUDA versions used during development; adapt these for your environment.

Results & artifacts
- Checkpoints (if present) are stored in `models_vA0/` (not committed to the repo by default; `.gitignore` excludes them).
- Example outputs produced during training live in `validation_images/`.

Future work
- Implement stochastic samplers (DDIM / ancestral sampling) and support temperature/eta control.
- Better training recipes: mixed precision, larger batch sizes, data augmentation and longer schedules.
- Replace/benchmark different text encoders and consider caching embeddings to speed up training.
- Add evaluation metrics (FID) and a clearer experiment tracking setup.

Contact
- This is a personal portfolio project to learn diffusion models. Use the code as a learning resource. Pull requests and suggestions are welcome.

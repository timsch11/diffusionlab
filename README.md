# Diffusion U-Net in Flax

An educational project: a Flax-based implementation of a conditional **Denoising Diffusion Probabilistic Model (DDPM)** with a diffusion U-Net, built from scratch.  
I did this project to dive deeper into **conditional diffusion models**, and I’ll share some of my insights here.  
I hope this might be useful for you if you are interested in diffusion models and want to get some practical experience :)

![grafik](https://github.com/timsch11/diffusionlab/blob/main/exampleimgs_small/t200/grinning_face.jpeg)

---

## 📌 Project Overview

The goal of this project was to **rebuild a text-conditioned diffusion model step by step**, without relying on large frameworks.  
This helped me really understand each moving part:

- Forward diffusion process
- Diffusion U-Net with ResNet blocks and attention
- Conditioning on text via cross-attention
- Training objective and noise schedule
- Sampling loop with classifier-free guidance (CFG)
- Practical concerns: standardization, dataset handling, masks

---

## 🔬 Key Components

### 1. Forward Diffusion (`forward.py`)
The forward process gradually adds Gaussian noise:

```math
x_t = \sqrt{\bar{\alpha}_t}\, x_0 \;+\; \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\quad \varepsilon \sim \mathcal{N}(0,I)
```

- Implemented a **cosine beta schedule** for stability.  
- Added utilities for `noisify` and reverse updates.

---

### 2. Diffusion U-Net (`encoder.py`, `decoder.py`, `bottleneck.py`, `timestamp_encoding.py`)

I build a 'template' block for each component:
- TimestampNet: Takes a timestep (integer) -> Creates sinusoidal encoding -> feds encoding through a linear layer and applies silu
  Purpose: Generates learned representation of timesteps, which are later fed into each block
- ResNet: input -> x -> norm(x) -> silu(x) -> conv(x) -> x + timestamp_embedding -> norm(x) -> silu(x) -> conv(x) -> x + input -> y
  Common ResNet architecture, I went with the norm-first approach as it showed to yield better results in recent research. Furthermore I chose 3x3 convolutions     mostly, as it showed to be a nice tradeoff of receptive field enhancement and computational expensiveness. The most special part is probably the timestamp        embedding which is fed into every ResNet to let the model know at what timestamp it operates, this is crucial for diffusion.
- Encoder: **ResNet 1** -> **Self-attention (optional)** -> **Cross-attention (optional)** -> **ResNet 2** -> **x_skip** -> **Conv**
  The last convolution reduces the resolution of the feature map and increases the channels by a certain factor. The tensor right before the downsampling takes     place is later fed into the corresponding decoder block as a skip connection. Why before? The decoder increases resolution before adding the skip connection,     therefore this is neccessary to make the resolutions match.
- Bottleneck: **ResNet 1** -> **Self-attention (optional)** -> **Cross-attention (optional)** -> **ResNet 2**
  No resolution change.
- Decoder: **Upsample (nearest neighbor)** -> **add conv(skip)** -> **ResNet 1** -> **Self-attention (optional)** -> **Cross-attention (optional)** -> **ResNet 2** -> **Conv**
  The nearest neighbor upsampling doubles resolution, then the skip connection is added, however first fed through a 1x1 convolution to make the channel sizes      match (neccessary) for addition. The last convolution reduces the channels.

The following diagram shows the model architecture, with the shape of the feature map at different stages.

      Input Image (Noise), [B, 64, 64, 3]
              | [B, 64, 64, 3]
              ▼
             Conv      Increases channels to base_dim
              | [B, 64, 64, 20]
        ┌─────▼─────┐      
        │  Encoder  │  Self-attention and cross-attention
        └─────┬─────┘
              │ [B, 32, 32, 40]
        ┌─────▼─────┐
        │  Encoder  │  Cross-attention
        └─────┬─────┘
              │ [B, 16, 16, 80]
        ┌─────▼─────┐
        │  Encoder  │  Self-attention and cross-attention
        └─────┬─────┘
              │ [B, 8, 8, 160]
        ┌─────▼─────┐
        │  Encoder  │  Self-attention
        └─────┬─────┘
              │ [B, 8, 8, 320]
        ┌─────▼─────┐
        │Bottleneck │  Self-attention and cross-attention
        └─────┬─────┘
              │ [B, 8, 8, 320]
        ┌─────▼─────┐
        │  Decoder  │  Self-attention
        └─────┬─────┘
              │  [B, 8, 8, 160]
        ┌─────▼─────┐
        │  Decoder  │  Self-attention and cross-attention
        └─────┬─────┘
              │  [B, 16, 16, 80]
        ┌─────▼─────┐
        │  Decoder  │  Cross-attention
        └─────┬─────┘
              │ [B, 32, 32, 40]
        ┌─────▼─────┐
        │  Decoder  │  Self-attention and cross-attention
        └─────┬─────┘
              | [B, 64, 64, 20]
              ▼
             Conv      Decrease channels back to 3 (RGB)
              | [B, 64, 64, 3]
              ▼
            Output


### 3. Text Embeddings (`prompt_embedding.py`)

- Started with **BGE-small** (retrieval model, 384d), however, the prompt had almost no influence on image generation (weak conditioning) -> I switched to clip-vit-base-patch32: larger model with much better embeddings for image generation tasks
- Produced both embeddings and **attention masks** for padding.

---

### 4. Data Handling (`dataloader.py`, `util.py`)

- Loads PNG images + prompts from CSV.  
- Applies **dataset-wide standardization** (per-channel mean/std) → fixes color bias (my early model produced very greenish images).  
- On-the-fly **noisification** and **classifier-free dropout** of embeddings.  
- Fully JIT-compatible batch builder with deterministic seeds.

---

### 5. Training (`train.py`)

- Loss: **MSE on predicted noise**
  ```math
  \mathcal{L} = \|\varepsilon_\theta(x_t, t, c) - \varepsilon\|^2
  ```
- Optimizer: **AdamW** with warmup + cosine decay.  
- Added gradient clipping for stability.  
- Logs training loss and periodic sample generations.

---

### 6. Sampling (`pipeline.py`)

Implements reverse diffusion from pure Gaussian noise.

- Deterministic sampler (DDIM-like, no stochastic noise).  
- **Classifier-free guidance (CFG)**:
- ```math
  \hat\varepsilon = \varepsilon_\text{uncond} + s \cdot \big(\varepsilon_\text{cond} - \varepsilon_\text{uncond}\big)
  ```
  where \(s\) is the guidance scale (typically 5–8).  
- Postprocess: invert normalization and save final images.

---

## ⚡ Insights & Pitfalls

- **Standardization**: per-channel dataset-wide mean/std normalization solved “green collapse”.  
- **Embeddings**: retrieval embeddings are too weak; CLIP works much better.  
- **Objective**: My first approach was to let the model predict the next image. However, training turned out to be very unstable, which is why I switched to ε-prediction (predict 'only' added noise at that timestep and construct image from that). Those approaches are mathmatically equivalent, however, ε-prediction seems to be more stable for deep-learning based approaches.
- **Feature map resolution**: I first downsampled to a resolution of 4x4 at the bottleneck, which was much too low. I experimented with different downsampling schemes and found that downsampling to 8x8 and keeping resolution unchanged at the most bottom encoder/decoder blocks was the sweetspot for this usecase (Higher resolutions -> very large increase in required VRAM, doubled resolution for one block -> roughly 4x memory demand for that block).
- CFG sampling significantly increased quality (more than I thought)

---

## 🧭 How to Use

### Training 

1. Prepare your dataset:
   - Install requirements ```uv pip install -r requirements.txt``` (or use plain pip)
   - Download dataset from https://www.kaggle.com/datasets/subinium/emojiimage-dataset/data
   - Optional: Experiment with hyperparameters in `params.py` (Note: changing BASE_DIM requires modifying the number of groups for the GroupedNorm inside of the model)
   - run `train.py`

### Generating your own images
1. Run `test.py`

**Note: This repository is exclusively for educational/research purposes, also see `LICENSE.md`**

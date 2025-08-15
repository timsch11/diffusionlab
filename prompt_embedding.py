from transformers import AutoTokenizer, FlaxAutoModel
import jax.numpy as jnp
from flax import nnx


# Model name
model_name = "BAAI/bge-small-en"

# Load tokenizer and Flax model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FlaxAutoModel.from_pretrained(model_name)



def embedd_prompts_seq(prompts, max_length=64):
    """
    Returns token-level embeddings for cross-attention:
      context: [B, T, D]  (D=384 for bge-small)
      attn_mask: [B, T]

    """

    if isinstance(prompts, str):
        prompts = [prompts]

    toks = tokenizer(
        prompts,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    outputs = model(**toks)

    # Per-token embeddings for cross-attention
    context = outputs.last_hidden_state  # [B, T, 384]
    attn_mask = jnp.asarray(toks["attention_mask"])  # [B, T]

    return context, attn_mask


def embedd_prompts_batched(prompts: list[str], chunk_size: int = 64) -> jnp.ndarray:
    """
    Embeds a list of prompts in smaller batches (to avoid OOM).
    
    Args:
        prompts (list[str]): List of text prompts to embed
        chunk_size (int): Number of prompts per chunk

    Returns:
        jnp.ndarray: Embeddings of shape (len(prompts), hidden_dim)
    """

    all_embeddings = []

    for i in range(0, len(prompts), chunk_size):
        batch_prompts = prompts[i:i + chunk_size]

        # Tokenize this batch
        inputs = tokenizer(batch_prompts, return_tensors="np", padding=True, truncation=True)

        # Run through model (Flax)
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        cls_embeddings = hidden[:, 0, :]    # (B, hidden_dim)

        # Normalize
        norm = jnp.linalg.norm(cls_embeddings, axis=-1, keepdims=True) + 1e-8
        normalized = cls_embeddings / norm

        all_embeddings.append(normalized)

    # Concatenate all embeddings into a single batch
    return jnp.concatenate(all_embeddings).reshape(-1, 1, 384)  # [B, 1, 384]


def embedd_prompt(prompt: str):
    inputs = tokenizer(prompt, return_tensors="np", padding=True, truncation=True)
    outputs = model(**inputs)
    
    # Extract [CLS] embedding
    last_hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
    cls_embedding = last_hidden_state[:, 0]        # shape: (1, hidden_dim)

    # Normalize
    norm = jnp.linalg.norm(cls_embedding, axis=-1, keepdims=True) + 1e-8
    cls_embedding = cls_embedding / norm

    return cls_embedding  # shape: (1, hidden_dim)



if __name__ == '__main__':
    out = embedd_prompts_seq([""])
    print(out[0].shape, out[1])
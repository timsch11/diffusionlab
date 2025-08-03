from transformers import AutoTokenizer, FlaxAutoModel
import jax.numpy as jnp
from flax import nnx


# Model name
model_name = "BAAI/bge-small-en"

# Load tokenizer and Flax model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FlaxAutoModel.from_pretrained(model_name)


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
    return jnp.concatenate(all_embeddings, axis=0)


@nnx.jit()
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
    inp = ["Smiling face", "Happy", "Smiling face", "Happy", "Smiling face", "Happy", "Smiling face", "Happy", "Smiling face", "Happy", "Smiling face", "Happy"]
    out = embedd_prompts_batched(inp)
    print(out.shape)
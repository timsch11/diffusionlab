from transformers import AutoTokenizer, FlaxCLIPTextModel
import jax.numpy as jnp


# Model name
model_name = "openai/clip-vit-base-patch32"

# Load tokenizer and Flax CLIP text model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FlaxCLIPTextModel.from_pretrained(model_name)



def embedd_prompts_seq(prompts, max_length=64):
    """
    Returns token-level embeddings for cross-attention:
      context: [B, T, D]  (D depends on model)
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

    # Convert only non-None numpy arrays to jax arrays for Flax model
    toks_jnp = {k: jnp.asarray(v) for k, v in toks.items() if v is not None}

    outputs = model(**toks_jnp)

    # Per-token embeddings for cross-attention
    context = outputs.last_hidden_state  # [B, T, D]
    attn_mask = jnp.asarray(toks["attention_mask"])  # [B, T]

    return context, attn_mask


if __name__ == '__main__':
    out = embedd_prompts_seq([""])
    print(out[0].shape, out[1])
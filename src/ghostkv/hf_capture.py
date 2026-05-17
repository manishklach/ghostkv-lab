"""Utilities for capturing real transformer Q/K tensors via HuggingFace models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from .bounds import compute_attention_upper_bound, eliminate_by_threshold, false_elimination_rate, topk_overlap
from .sketches import (
    cosine_rank_correlation,
    project_keys,
    project_query,
    random_projection_matrix,
    sketch_similarity,
)


FALLBACK_MODEL_NAME = "gpt2"


@dataclass
class AttentionCaptureResult:
    """Container for captured query/key tensors from a transformer layer."""

    model_name: str
    prompt: str
    layer_idx: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tokens: list[str]
    query_states: torch.Tensor
    key_states: torch.Tensor


def _set_torch_seed(seed: int) -> None:
    """Set deterministic seeds for PyTorch CPU inference."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    """Load a lightweight causal LM and tokenizer on CPU, with GPT-2 fallback."""
    _set_torch_seed(0)
    requested_name = model_name
    resolved_name = requested_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(requested_name)
        model = AutoModelForCausalLM.from_pretrained(requested_name)
    except Exception:
        resolved_name = FALLBACK_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        model = AutoModelForCausalLM.from_pretrained(resolved_name)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to("cpu")
    model.eval()
    setattr(model.config, "ghostkv_requested_model_name", requested_name)
    setattr(model.config, "ghostkv_resolved_model_name", resolved_name)
    return model, tokenizer


def _decoder_layers(model: Any) -> list[Any]:
    """Return decoder layers from GPT-2- or Llama-style causal LMs."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise TypeError("Unsupported model architecture for GhostKV attention capture.")


def _attention_module(model: Any, layer_idx: int) -> Any:
    """Return the attention module for a decoder layer."""
    layers = _decoder_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"layer_idx {layer_idx} out of range for {len(layers)} layers.")

    layer = layers[layer_idx]
    if hasattr(layer, "attn"):
        return layer.attn
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    raise TypeError("Unsupported attention module layout.")


def _max_length_for_tokenizer(tokenizer: Any) -> int:
    """Choose a practical maximum prompt length for CPU-side evaluation."""
    model_max_length = getattr(tokenizer, "model_max_length", 256)
    if model_max_length is None or model_max_length > 4096:
        return 256
    return int(min(model_max_length, 256))


def _capture_gpt2_qk(attention_module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture GPT-2 query/key tensors from hidden states."""
    query_states, key_states, _ = attention_module.c_attn(hidden_states).split(
        attention_module.split_size,
        dim=2,
    )
    query_shape = (*query_states.shape[:-1], -1, attention_module.head_dim)
    key_shape = (*key_states.shape[:-1], -1, attention_module.head_dim)
    query_states = query_states.view(query_shape).transpose(1, 2)
    key_states = key_states.view(key_shape).transpose(1, 2)
    return query_states, key_states


def _capture_llama_qk(model: Any, attention_module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture Llama-family query/key tensors including rotary embedding."""
    batch_size, seq_len, _ = hidden_states.shape
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
    hidden_shape = (*hidden_states.shape[:-1], -1, attention_module.head_dim)

    query_states = attention_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attention_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    rotary_module = model.model.rotary_emb
    cos, sin = rotary_module(hidden_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if key_states.shape[1] != query_states.shape[1]:
        repeat_factor = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
    return query_states, key_states


def capture_qk_tensors(model: Any, tokenizer: Any, prompt: str, layer_idx: int) -> AttentionCaptureResult:
    """Run a prompt through the model and extract layer-local Q/K tensors."""
    max_length = _max_length_for_tokenizer(tokenizer)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to("cpu")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states[layer_idx]
    attention_module = _attention_module(model, layer_idx)
    with torch.no_grad():
        if hasattr(attention_module, "c_attn"):
            query_states, key_states = _capture_gpt2_qk(attention_module, hidden_states)
        elif hasattr(attention_module, "q_proj") and hasattr(attention_module, "k_proj"):
            query_states, key_states = _capture_llama_qk(model, attention_module, hidden_states)
        else:
            raise TypeError("Unsupported attention module for Q/K capture.")

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return AttentionCaptureResult(
        model_name=str(getattr(model.config, "ghostkv_resolved_model_name", model.config._name_or_path)),
        prompt=prompt,
        layer_idx=layer_idx,
        input_ids=input_ids.cpu(),
        attention_mask=attention_mask.cpu(),
        tokens=tokens,
        query_states=query_states.cpu(),
        key_states=key_states.cpu(),
    )


def compute_exact_attention_scores(query_states: np.ndarray, key_states: np.ndarray) -> np.ndarray:
    """Compute scaled dot-product attention scores."""
    if query_states.ndim == 1 and key_states.ndim == 2:
        return (key_states @ query_states) / np.sqrt(query_states.shape[-1])
    if query_states.ndim == 2 and key_states.ndim == 3:
        return np.einsum("hd,hsd->hs", query_states, key_states) / np.sqrt(query_states.shape[-1])
    raise ValueError("Unsupported query/key shapes for exact attention score computation.")


def flatten_attention_heads(
    query_states: torch.Tensor | np.ndarray,
    key_states: torch.Tensor | np.ndarray,
    query_position: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten batch dimension and select a query position for per-head analysis."""
    if isinstance(query_states, torch.Tensor):
        query_states = query_states.detach().cpu().numpy()
    if isinstance(key_states, torch.Tensor):
        key_states = key_states.detach().cpu().numpy()

    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("Expected query_states and key_states with shape [batch, heads, seq, dim].")

    selected_query = query_states[0, :, query_position, :]
    selected_keys = key_states[0]
    return selected_query, selected_keys


def extract_attention_statistics(
    query_states: torch.Tensor | np.ndarray,
    key_states: torch.Tensor | np.ndarray,
    sketch_dim: int,
    theta: float,
    epsilon: float,
    sigma: float,
    topk: int = 32,
    seed: int = 0,
    query_position: int = -1,
) -> dict[str, Any]:
    """Compute per-head and aggregate GhostKV-style metrics on real attention tensors."""
    query_heads, key_heads = flatten_attention_heads(query_states, key_states, query_position=query_position)
    projection_matrix = random_projection_matrix(query_heads.shape[-1], sketch_dim, seed=seed)

    per_head_rows: list[dict[str, float | int]] = []
    for head_idx in range(query_heads.shape[0]):
        exact_scores = compute_exact_attention_scores(query_heads[head_idx], key_heads[head_idx])
        key_sketches = project_keys(key_heads[head_idx], projection_matrix)
        query_sketch = project_query(query_heads[head_idx], projection_matrix)
        sketch_scores = sketch_similarity(query_sketch, key_sketches)
        bounds = compute_attention_upper_bound(sketch_scores, epsilon, sigma)
        eliminated_mask = eliminate_by_threshold(bounds, theta)
        effective_k = min(topk, exact_scores.shape[0])
        true_threshold = np.partition(exact_scores, -effective_k)[-effective_k]

        per_head_rows.append(
            {
                "head_idx": head_idx,
                "sketch_dim": sketch_dim,
                "theta": theta,
                "topk_overlap": float(topk_overlap(exact_scores, sketch_scores, effective_k)),
                "rank_correlation": float(cosine_rank_correlation(exact_scores, sketch_scores)),
                "false_elimination_rate": float(
                    false_elimination_rate(exact_scores, eliminated_mask, true_threshold)
                ),
                "elimination_rate": float(np.mean(eliminated_mask)),
                "num_tokens": int(exact_scores.shape[0]),
            }
        )

    aggregate = {
        "sketch_dim": sketch_dim,
        "theta": theta,
        "topk_overlap_mean": float(np.mean([row["topk_overlap"] for row in per_head_rows])),
        "topk_overlap_std": float(np.std([row["topk_overlap"] for row in per_head_rows])),
        "rank_correlation_mean": float(np.mean([row["rank_correlation"] for row in per_head_rows])),
        "false_elimination_rate_mean": float(
            np.mean([row["false_elimination_rate"] for row in per_head_rows])
        ),
        "elimination_rate_mean": float(np.mean([row["elimination_rate"] for row in per_head_rows])),
        "num_heads": int(len(per_head_rows)),
    }
    return {"per_head_rows": per_head_rows, "aggregate": aggregate}

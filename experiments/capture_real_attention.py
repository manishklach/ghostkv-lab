# GhostKV Lab — github.com/manishklach/ghostkv-lab
# Patent: IN 202641062451
"""Capture real Q/K tensors from modern transformer attention modules."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from huggingface_hub import login
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from ghostkv.sketches import random_projection_matrix


LOGGER = logging.getLogger("ghostkv.capture_real_attention")
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG_PATH = REPO_ROOT / ".ghostkv_env_config.json"
REAL_ATTENTION_DIR = REPO_ROOT / "data" / "real_attention"
RESULTS_DIR = REPO_ROOT / "results" / "real_attention_modern"
PROMPTS = [
    "The attention mechanism in transformers allows each token to",
    "In a long document about climate change, the key finding was that",
    "The capital of France is Paris. The Eiffel Tower was built in",
]
DEFAULT_CONTEXT_LENGTHS = [512, 1024, 2048]
DEFAULT_SKETCH_DIMS = [16, 32, 64, 128]
DEFAULT_THETAS = [0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_SAVE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]
DEFAULT_SAVE_HEADS = [0, 4, 8, 12]
MAX_SPEARMAN_POINTS = 20_000
FALLBACK_MODEL_ID = "gpt2"


@dataclass
class LoadedModelBundle:
    """Bundle together model, tokenizer, and resolved loading metadata."""

    model_id: str
    requested_model_id: str
    tokenizer: Any
    model: Any
    device: str
    actual_dtype: str
    load_in_4bit: bool


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _login_if_needed() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            login(token=token, add_to_git_credential=False)
        except Exception as exc:  # pragma: no cover - depends on local HF state
            LOGGER.warning("Hugging Face login failed: %s", exc)


def _load_env_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Could not parse %s; ignoring environment config.", path)
        return {}


def _detect_available_vram_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        del free_bytes
        return round(total_bytes / (1024.0**3), 2)
    except Exception:  # pragma: no cover - CUDA state varies by environment
        return None


def _detect_available_ram_gb() -> float:
    return round(psutil.virtual_memory().available / (1024.0**3), 2)


def _recommended_load_kwargs(device: str, precision: str, load_in_4bit: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if device == "cuda" and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning("bitsandbytes is unavailable; disabling 4-bit loading: %s", exc)
    elif device == "cuda" and precision == "fp16":
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float32
    return kwargs


def _memory_safe_model_choice(requested_model_id: str, requested_device: str, requested_precision: str) -> tuple[str, str, str]:
    ram_gb = _detect_available_ram_gb()
    vram_gb = _detect_available_vram_gb()
    if requested_device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable; falling back from %s to CPU-friendly GPT-2.", requested_model_id)
        return FALLBACK_MODEL_ID, "cpu", "fp32"
    if requested_device == "cuda" and vram_gb is not None and vram_gb < 6.0:
        LOGGER.warning("Available VRAM %.2f GB is too small for %s; using GPT-2.", vram_gb, requested_model_id)
        return FALLBACK_MODEL_ID, "cpu", "fp32"
    if ram_gb < 6.0 and requested_model_id != FALLBACK_MODEL_ID:
        LOGGER.warning("Available RAM %.2f GB is low; using GPT-2.", ram_gb)
        return FALLBACK_MODEL_ID, "cpu", "fp32"
    return requested_model_id, requested_device, requested_precision


def load_modern_model_bundle(
    requested_model_id: str | None,
    config_path: Path,
) -> LoadedModelBundle:
    """Load a Hugging Face model with env-driven recommendation and safe fallback."""
    _login_if_needed()
    env_config = _load_env_config(config_path)
    base_model_id = requested_model_id or str(env_config.get("model_id") or env_config.get("recommended_model_id") or FALLBACK_MODEL_ID)
    requested_device = str(env_config.get("device") or env_config.get("recommended_device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    requested_precision = str(env_config.get("precision") or env_config.get("recommended_precision") or ("fp16" if requested_device == "cuda" else "fp32"))
    requested_load_in_4bit = bool(env_config.get("load_in_4bit", False))

    model_id, device, precision = _memory_safe_model_choice(base_model_id, requested_device, requested_precision)
    load_kwargs = _recommended_load_kwargs(device, precision, requested_load_in_4bit)
    if model_id == FALLBACK_MODEL_ID:
        load_kwargs.pop("quantization_config", None)
        load_kwargs["dtype"] = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            output_attentions=False,
            output_hidden_states=False,
            **load_kwargs,
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to load {model_id}: {exc}. Falling back to {FALLBACK_MODEL_ID}.",
            stacklevel=2,
        )
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL_ID,
            output_attentions=False,
            output_hidden_states=False,
            dtype=torch.float32,
        )
        model_id = FALLBACK_MODEL_ID
        device = "cpu"
        precision = "fp32"
        requested_load_in_4bit = False

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cpu":
        model.to("cpu")
    model.eval()

    actual_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    return LoadedModelBundle(
        model_id=model_id,
        requested_model_id=base_model_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
        actual_dtype=actual_dtype,
        load_in_4bit=requested_load_in_4bit and model_id != FALLBACK_MODEL_ID,
    )


def _model_type(model: Any) -> str:
    return str(getattr(model.config, "model_type", "")).lower()


def _decoder_layers(model: Any) -> list[Any]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise TypeError("Unsupported model architecture for GhostKV capture.")


def _max_model_length(model: Any, tokenizer: Any) -> int:
    model_limits = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    valid = [int(limit) for limit in model_limits if isinstance(limit, int) and 0 < limit < 100_000]
    if valid:
        return min(valid)
    return 1024


def _prepare_inputs(tokenizer: Any, prompt: str, requested_context_length: int, model_max_length: int, device: str) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        prompt_ids = [eos_token_id]

    effective_length = min(requested_context_length, model_max_length)
    repeats = (effective_length + len(prompt_ids) - 1) // len(prompt_ids)
    repeated_ids = (prompt_ids * repeats)[:effective_length]
    input_ids = torch.tensor([repeated_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "requested_context_length": torch.tensor(requested_context_length, dtype=torch.long),
        "effective_context_length": torch.tensor(effective_length, dtype=torch.long),
    }


def _reshape_heads(tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    batch_size, seq_len, _ = tensor.shape
    return tensor.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _split_gpt2_qkv(qkv: torch.Tensor, module: Any) -> tuple[torch.Tensor, torch.Tensor]:
    query_states, key_states, _ = qkv.split(module.split_size, dim=2)
    return (
        _reshape_heads(query_states, module.num_heads, module.head_dim),
        _reshape_heads(key_states, module.num_heads, module.head_dim),
    )


def _split_gpt_neox_qkv(qkv: torch.Tensor, module: Any) -> tuple[torch.Tensor, torch.Tensor]:
    num_heads = int(module.num_attention_heads)
    head_dim = int(module.head_size)
    batch_size, seq_len, _ = qkv.shape
    qkv = qkv.view(batch_size, seq_len, num_heads, 3 * head_dim)
    query_states, key_states, _ = torch.split(qkv, head_dim, dim=-1)
    return (
        query_states.permute(0, 2, 1, 3).contiguous(),
        key_states.permute(0, 2, 1, 3).contiguous(),
    )


def _reshape_llama_qk(query_linear: torch.Tensor, key_linear: torch.Tensor, module: Any) -> tuple[torch.Tensor, torch.Tensor]:
    num_heads = int(module.num_heads)
    num_kv_heads = int(getattr(module, "num_key_value_heads", num_heads))
    head_dim = int(module.head_dim)
    query_states = _reshape_heads(query_linear, num_heads, head_dim)
    key_states = _reshape_heads(key_linear, num_kv_heads, head_dim)
    if num_kv_heads != num_heads:
        repeat_factor = num_heads // num_kv_heads
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
    return query_states, key_states


def capture_attention_qk(
    bundle: LoadedModelBundle,
    prompt: str,
    requested_context_length: int,
    selected_layers: list[int] | None = None,
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], int, int]:
    """Capture raw attention Q/K tensors for selected decoder layers."""
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = "cuda" if bundle.device == "cuda" and torch.cuda.is_available() else "cpu"
    model_type = _model_type(model)
    layers = _decoder_layers(model)
    layer_indices = selected_layers or list(range(len(layers)))
    max_length = _max_model_length(model, tokenizer)
    model_inputs = _prepare_inputs(tokenizer, prompt, requested_context_length, max_length, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    effective_context_length = int(model_inputs["effective_context_length"].item())

    captured: dict[int, dict[str, torch.Tensor]] = {}
    handles: list[Any] = []

    def register_tensor(layer_idx: int, name: str) -> Any:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            captured.setdefault(layer_idx, {})[name] = output.detach().cpu()

        return hook

    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        layer = layers[layer_idx]
        if model_type == "gpt2" and hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
            handles.append(layer.attn.c_attn.register_forward_hook(register_tensor(layer_idx, "qkv")))
        elif "gpt_neox" in model_type and hasattr(layer, "attention") and hasattr(layer.attention, "query_key_value"):
            handles.append(layer.attention.query_key_value.register_forward_hook(register_tensor(layer_idx, "qkv")))
        elif hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj") and hasattr(layer.self_attn, "k_proj"):
            handles.append(layer.self_attn.q_proj.register_forward_hook(register_tensor(layer_idx, "q_linear")))
            handles.append(layer.self_attn.k_proj.register_forward_hook(register_tensor(layer_idx, "k_linear")))
        else:
            LOGGER.warning("Skipping unsupported layer %s for model_type=%s.", layer_idx, model_type)

    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    qk_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_idx in layer_indices:
        if layer_idx not in captured:
            continue
        layer = layers[layer_idx]
        if "qkv" in captured[layer_idx]:
            qkv = captured[layer_idx]["qkv"]
            if model_type == "gpt2":
                qk_by_layer[layer_idx] = _split_gpt2_qkv(qkv, layer.attn)
            elif "gpt_neox" in model_type:
                qk_by_layer[layer_idx] = _split_gpt_neox_qkv(qkv, layer.attention)
        elif "q_linear" in captured[layer_idx] and "k_linear" in captured[layer_idx]:
            qk_by_layer[layer_idx] = _reshape_llama_qk(
                captured[layer_idx]["q_linear"],
                captured[layer_idx]["k_linear"],
                layer.self_attn,
            )
    return qk_by_layer, requested_context_length, effective_context_length


def _compute_exact_scores(query_states: np.ndarray, key_states: np.ndarray) -> np.ndarray:
    head_dim = query_states.shape[-1]
    return (query_states @ key_states.T) / math.sqrt(head_dim)


def _compute_sketch_scores(query_states: np.ndarray, key_states: np.ndarray, sketch_dim: int, seed: int) -> np.ndarray:
    projection = random_projection_matrix(query_states.shape[-1], sketch_dim, seed=seed)
    query_sketch = query_states @ projection
    key_sketch = key_states @ projection
    return (query_sketch @ key_sketch.T) / math.sqrt(sketch_dim)


def _topk_overlap(flat_exact: np.ndarray, flat_sketch: np.ndarray, k: int) -> float:
    effective_k = min(k, flat_exact.size)
    if effective_k == 0:
        return 0.0
    exact_top = set(np.argpartition(flat_exact, -effective_k)[-effective_k:].tolist())
    sketch_top = set(np.argpartition(flat_sketch, -effective_k)[-effective_k:].tolist())
    return float(len(exact_top.intersection(sketch_top)) / effective_k)


def _sampled_spearman(exact_scores: np.ndarray, sketch_scores: np.ndarray, seed: int) -> tuple[float, int]:
    flat_exact = exact_scores.reshape(-1)
    flat_sketch = sketch_scores.reshape(-1)
    sample_size = min(MAX_SPEARMAN_POINTS, flat_exact.size)
    if sample_size < flat_exact.size:
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(flat_exact.size, size=sample_size, replace=False)
        flat_exact = flat_exact[sample_indices]
        flat_sketch = flat_sketch[sample_indices]
    correlation = spearmanr(flat_exact, flat_sketch).correlation
    if correlation is None or np.isnan(correlation):
        return 0.0, sample_size
    return float(correlation), sample_size


def _metrics_for_head(
    exact_scores: np.ndarray,
    sketch_scores: np.ndarray,
    theta: float,
) -> dict[str, float | int]:
    sketch_max = sketch_scores.max(axis=-1)
    exact_max = exact_scores.max(axis=-1)
    eliminated = sketch_max < theta
    eliminated_count = int(eliminated.sum())
    false_eliminated = np.logical_and(eliminated, exact_max > theta)
    false_eliminated_count = int(false_eliminated.sum())
    false_rate = (
        float(false_eliminated_count / eliminated_count)
        if eliminated_count > 0
        else 0.0
    )
    elimination_rate = float(np.mean(eliminated))
    return {
        "false_elimination_rate": false_rate,
        "elimination_rate": elimination_rate,
        "resurrection_rate": float(1.0 - elimination_rate),
        "eliminated_count": eliminated_count,
        "false_eliminated_count": false_eliminated_count,
        "total_tokens": int(exact_scores.shape[0]),
        "survivor_count": int(exact_scores.shape[0] - eliminated_count),
    }


def _sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace("\\", "__")


def _summarize_metrics(dataframe: pd.DataFrame, bundle: LoadedModelBundle) -> dict[str, Any]:
    summary = {
        "requested_model_id": bundle.requested_model_id,
        "resolved_model_id": bundle.model_id,
        "device": bundle.device,
        "actual_dtype": bundle.actual_dtype,
        "load_in_4bit": bundle.load_in_4bit,
        "num_rows": int(len(dataframe)),
        "mean_false_elimination_rate": float(dataframe["false_elimination_rate"].mean()),
        "mean_elimination_rate": float(dataframe["elimination_rate"].mean()),
        "mean_rank_correlation": float(dataframe["rank_correlation"].mean()),
        "mean_top8_overlap": float(dataframe["top8_overlap"].mean()),
        "mean_top16_overlap": float(dataframe["top16_overlap"].mean()),
        "mean_top32_overlap": float(dataframe["top32_overlap"].mean()),
        "layers_evaluated": sorted({int(value) for value in dataframe["layer_idx"].unique().tolist()}),
        "heads_evaluated": sorted({int(value) for value in dataframe["head_idx"].unique().tolist()}),
        "contexts_evaluated": sorted({int(value) for value in dataframe["effective_context_length"].unique().tolist()}),
    }
    return summary


def run_capture(args: argparse.Namespace) -> dict[str, Path]:
    """Run the modern real-attention capture experiment."""
    _set_seed(args.seed)
    bundle = load_modern_model_bundle(args.model_id, ENV_CONFIG_PATH)
    layer_filter = _parse_int_list(args.layers) if args.layers else None
    head_filter = _parse_int_list(args.heads) if args.heads else None
    context_lengths = _parse_int_list(args.context_lengths)
    sketch_dims = _parse_int_list(args.sketch_dims)
    thetas = _parse_float_list(args.thetas)
    save_layers = set(_parse_int_list(args.save_layers))
    save_heads = set(_parse_int_list(args.save_heads))

    REAL_ATTENTION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    layers = _decoder_layers(bundle.model)
    if layer_filter is None:
        layer_filter = [layer_idx for layer_idx in DEFAULT_SAVE_LAYERS if layer_idx < len(layers)]
        if not layer_filter:
            if len(layers) >= 4:
                layer_filter = sorted({0, len(layers) // 4, len(layers) // 2, (3 * len(layers)) // 4})
            else:
                layer_filter = list(range(len(layers)))
    layer_filter = [layer_idx for layer_idx in layer_filter if 0 <= layer_idx < len(layers)]

    rows: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        for requested_context_length in context_lengths:
            qk_by_layer, requested_len, effective_len = capture_attention_qk(
                bundle,
                prompt,
                requested_context_length=requested_context_length,
                selected_layers=layer_filter,
            )
            for layer_idx, (query_states_torch, key_states_torch) in qk_by_layer.items():
                query_states = query_states_torch[0].detach().cpu().numpy().astype(np.float32)
                key_states = key_states_torch[0].detach().cpu().numpy().astype(np.float32)
                selected_heads = (
                    [head_idx for head_idx in head_filter if 0 <= head_idx < query_states.shape[0]]
                    if head_filter
                    else list(range(query_states.shape[0]))
                )
                for head_idx in selected_heads:
                    if layer_idx in save_layers and head_idx in save_heads:
                        raw_tensor_path = REAL_ATTENTION_DIR / (
                            f"{_sanitize_model_id(bundle.model_id)}_{effective_len}_layer{layer_idx}_head{head_idx}.npz"
                        )
                        np.savez_compressed(
                            raw_tensor_path,
                            q=query_states[head_idx].astype(np.float16),
                            k=key_states[head_idx].astype(np.float16),
                            model_id=bundle.model_id,
                            prompt=prompt,
                            requested_context_length=requested_len,
                            effective_context_length=effective_len,
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                        )

                    exact_scores = _compute_exact_scores(query_states[head_idx], key_states[head_idx])
                    flat_exact = exact_scores.reshape(-1)

                    for sketch_dim in sketch_dims:
                        sketch_seed = args.seed + prompt_idx * 10_000 + requested_context_length * 10 + layer_idx * 100 + head_idx * 1_000 + sketch_dim
                        sketch_scores = _compute_sketch_scores(
                            query_states[head_idx],
                            key_states[head_idx],
                            sketch_dim=sketch_dim,
                            seed=sketch_seed,
                        )
                        flat_sketch = sketch_scores.reshape(-1)
                        rank_corr, sample_size = _sampled_spearman(exact_scores, sketch_scores, seed=sketch_seed)
                        top8_overlap = _topk_overlap(flat_exact, flat_sketch, 8)
                        top16_overlap = _topk_overlap(flat_exact, flat_sketch, 16)
                        top32_overlap = _topk_overlap(flat_exact, flat_sketch, 32)

                        for theta in thetas:
                            metrics = _metrics_for_head(exact_scores, sketch_scores, theta)
                            rows.append(
                                {
                                    "model_id": bundle.model_id,
                                    "requested_model_id": bundle.requested_model_id,
                                    "device": bundle.device,
                                    "actual_dtype": bundle.actual_dtype,
                                    "prompt_idx": prompt_idx,
                                    "prompt_text": prompt,
                                    "requested_context_length": requested_len,
                                    "effective_context_length": effective_len,
                                    "layer_idx": layer_idx,
                                    "head_idx": head_idx,
                                    "sketch_dim": sketch_dim,
                                    "theta": theta,
                                    "false_elimination_rate": metrics["false_elimination_rate"],
                                    "elimination_rate": metrics["elimination_rate"],
                                    "resurrection_rate": metrics["resurrection_rate"],
                                    "top8_overlap": top8_overlap,
                                    "top16_overlap": top16_overlap,
                                    "top32_overlap": top32_overlap,
                                    "topk_overlap": top32_overlap,
                                    "rank_correlation": rank_corr,
                                    "rank_corr_sample_size": sample_size,
                                    "eliminated_count": metrics["eliminated_count"],
                                    "false_eliminated_count": metrics["false_eliminated_count"],
                                    "total_tokens": metrics["total_tokens"],
                                    "survivor_count": metrics["survivor_count"],
                                }
                            )

    dataframe = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "metrics.csv"
    summary_path = RESULTS_DIR / "summary.json"
    dataframe.to_csv(metrics_path, index=False)
    summary_path.write_text(json.dumps(_summarize_metrics(dataframe, bundle), indent=2), encoding="utf-8")
    return {"metrics_csv": metrics_path, "summary_json": summary_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--layers", type=str, default="")
    parser.add_argument("--heads", type=str, default="0,4,8,12")
    parser.add_argument("--context-lengths", type=str, default="512,1024,2048")
    parser.add_argument("--sketch-dims", type=str, default="16,32,64,128")
    parser.add_argument("--thetas", type=str, default="0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--save-layers", type=str, default="0,4,8,12,16,20,24,28")
    parser.add_argument("--save-heads", type=str, default="0,4,8,12")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    artifacts = run_capture(args)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

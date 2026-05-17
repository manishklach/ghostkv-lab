"""GhostKV synthetic evaluation toolkit."""

from .bandwidth import (
    bandwidth_reduction_ratio,
    ghostkv_bytes,
    quantized_kv_bytes,
    standard_kv_bytes,
)
from .hf_capture import (
    capture_qk_tensors,
    compute_exact_attention_scores,
    extract_attention_statistics,
    flatten_attention_heads,
    load_model_and_tokenizer,
)
from .hierarchical import (
    coarse_elimination,
    hierarchical_token_elimination,
    simple_anchor_clustering,
    token_level_elimination,
)
from .bounds import (
    compute_attention_upper_bound,
    eliminate_by_threshold,
    false_elimination_rate,
    topk_overlap,
)
from .lifecycle import ARCHIVE, GHOST, HOT, WARM, GhostRecord, ghostify_tokens, transition_states
from .plotting import (
    plot_resurrection_rate_vs_bandwidth,
    plot_sketch_dim_vs_topk_overlap,
    plot_theta_vs_elimination_rate,
)
from .simulator import SyntheticGhostKVSimulator

__all__ = [
    "ARCHIVE",
    "GHOST",
    "HOT",
    "WARM",
    "GhostRecord",
    "SyntheticGhostKVSimulator",
    "bandwidth_reduction_ratio",
    "compute_attention_upper_bound",
    "capture_qk_tensors",
    "coarse_elimination",
    "compute_exact_attention_scores",
    "eliminate_by_threshold",
    "extract_attention_statistics",
    "false_elimination_rate",
    "flatten_attention_heads",
    "ghostify_tokens",
    "ghostkv_bytes",
    "hierarchical_token_elimination",
    "load_model_and_tokenizer",
    "plot_resurrection_rate_vs_bandwidth",
    "plot_sketch_dim_vs_topk_overlap",
    "plot_theta_vs_elimination_rate",
    "quantized_kv_bytes",
    "simple_anchor_clustering",
    "standard_kv_bytes",
    "token_level_elimination",
    "topk_overlap",
    "transition_states",
]

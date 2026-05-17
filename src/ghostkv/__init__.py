"""GhostKV synthetic evaluation toolkit."""

from .bandwidth import (
    bandwidth_reduction_ratio,
    ghostkv_bytes,
    quantized_kv_bytes,
    standard_kv_bytes,
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
    "eliminate_by_threshold",
    "false_elimination_rate",
    "ghostify_tokens",
    "ghostkv_bytes",
    "plot_resurrection_rate_vs_bandwidth",
    "plot_sketch_dim_vs_topk_overlap",
    "plot_theta_vs_elimination_rate",
    "quantized_kv_bytes",
    "standard_kv_bytes",
    "topk_overlap",
    "transition_states",
]

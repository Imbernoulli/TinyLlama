"""Positional encoding registry.

Each positional encoding provides two functions:
    - build_cache(seq_len, n_elem, dtype, device, base, condense_ratio) -> (cos, sin)
    - apply(q, k, cos, sin) -> (q, k)

To add a custom positional encoding:
    1. Define build_cache and apply functions with the signatures above
    2. Add them to POS_ENCODING_REGISTRY as {"build_cache": ..., "apply": ...}
    3. Set _pos_encoding="my_encoding" in your Config
"""
import torch
from typing import Tuple

RoPECache = Tuple[torch.Tensor, torch.Tensor]


def _cast_cache(cos: torch.Tensor, sin: torch.Tensor, dtype: torch.dtype) -> RoPECache:
    """Cast cos/sin to appropriate dtype for fused rotary embedding."""
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


# ---- Standard RoPE ----

def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Standard Rotary Position Embedding.

    theta_i = base^(-2(i-1)/d), i in [1, ..., d/2]
    Change `base` (e.g. 10000 -> 1000000) for NTK-Aware RoPE scaling.
    Change `condense_ratio` (e.g. 4) for compressed positional indexing.
    """
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta)
    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    return _cast_cache(cos, sin, dtype)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key. Uses fused CUDA kernel when available."""
    from .fused_rotary_embedding import apply_rotary_emb_func

    q = apply_rotary_emb_func(q, cos, sin, False, True)
    k = apply_rotary_emb_func(k, cos, sin, False, True)
    return q, k


# ---- Dynamic NTK RoPE ----

def build_dynamic_ntk_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Dynamic NTK-Aware RoPE.

    Dynamically scales base frequency based on actual sequence length.
    Automatically adjusts for sequences longer than the training length.

    Assumes training length was 2048. For sequences > 2048, applies NTK scaling.
    Formula: base_new = base × (seq_len/train_len)^(d/(d-2))
    """
    train_seq_len = 2048  # Default training sequence length
    if seq_len > train_seq_len:
        # Dynamic scaling when sequence exceeds training length
        scale = seq_len / train_seq_len
        base = int(base * (scale ** (n_elem / (n_elem - 2))))

    return build_rope_cache(seq_len, n_elem, dtype, device, base, condense_ratio)


def apply_dynamic_ntk_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as standard RoPE application."""
    return apply_rope(q, k, cos, sin)


# ---- YaRN (Yet another RoPE extensioN) ----

def build_yarn_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """YaRN: Temperature-scaled RoPE with attention ramp.

    Combines NTK-style scaling with temperature ramping for different frequency bands.
    High-frequency components use less scaling, low-frequency use more.

    Reference: https://arxiv.org/abs/2309.00071
    """
    train_seq_len = 2048
    scale = max(seq_len / train_seq_len, 1.0)

    # YaRN parameters
    alpha = scale ** (n_elem / (n_elem - 2))  # NTK-style base scaling
    beta_fast = 32  # Wavelength threshold for high freq
    beta_slow = 1   # Wavelength threshold for low freq

    # Compute base frequencies
    dim_indices = torch.arange(0, n_elem, 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (dim_indices / n_elem))

    # Apply temperature ramp
    wavelengths = 2 * torch.pi / freqs
    ramp = torch.clamp((wavelengths - beta_fast) / (beta_slow - beta_fast), 0, 1)
    scaled_freqs = freqs / (ramp * (alpha - 1) + 1)

    # Build cache
    seq_idx = torch.arange(seq_len, device=device, dtype=torch.float32) / condense_ratio
    idx_theta = torch.outer(seq_idx, scaled_freqs)
    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    return _cast_cache(cos, sin, dtype)


def apply_yarn_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as standard RoPE application."""
    return apply_rope(q, k, cos, sin)


# ---- Linear Scaled RoPE (Position Interpolation) ----

def build_linear_scaled_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Linear position interpolation (PI) for RoPE.

    Compresses positions linearly into the training range via condense_ratio.
    For 4x extension, set condense_ratio=4.

    This is already handled by the standard rope with condense_ratio,
    but provided as explicit variant for clarity.
    """
    return build_rope_cache(seq_len, n_elem, dtype, device, base, condense_ratio)


def apply_linear_scaled_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same as standard RoPE application."""
    return apply_rope(q, k, cos, sin)


# ---- No Positional Encoding (NoPE) ----

def build_none_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    """Dummy cache for no positional encoding. Returns minimal-shaped tensors."""
    half = max(n_elem // 2, 1)
    cos = torch.ones(seq_len, half, device=device)
    sin = torch.zeros(seq_len, half, device=device)
    return _cast_cache(cos, sin, dtype)


def apply_none(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """No-op: return q, k unchanged."""
    return q, k


# ---- Registry ----

POS_ENCODING_REGISTRY = {
    "rope": {"build_cache": build_rope_cache, "apply": apply_rope},
    "dynamic_ntk": {"build_cache": build_dynamic_ntk_rope_cache, "apply": apply_dynamic_ntk_rope},
    "yarn": {"build_cache": build_yarn_rope_cache, "apply": apply_yarn_rope},
    "linear_scaled": {"build_cache": build_linear_scaled_rope_cache, "apply": apply_linear_scaled_rope},
    "none": {"build_cache": build_none_cache, "apply": apply_none},
}


def get_pos_encoding(name: str) -> dict:
    """Get positional encoding by name."""
    if name not in POS_ENCODING_REGISTRY:
        raise ValueError(
            f"Unknown positional encoding '{name}'. "
            f"Available: {list(POS_ENCODING_REGISTRY.keys())}"
        )
    return POS_ENCODING_REGISTRY[name]

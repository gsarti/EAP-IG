"""GWAI: Gradient-Weighted ALTI Interactions for circuit discovery.

Combines structural decomposition (which source vectors contribute to the
residual stream) with task-specific gradients (which directions in the
residual stream matter for the metric). GIM-corrected JVPs account for
self-repair effects by propagating contributions through subsequent layers.

Two scoring modes:
  - 'gradient': score = ||z_j^(k) * grad||_1  (gradient-weighted contribution)
  - 'proximity': score = ALTI proximity metric (purely geometric, task-agnostic)

Incremental propagation: sources are propagated through each layer once and
scored at each destination along the way, making cost independent of k.
"""

import math
from typing import Optional, Literal

import torch
from torch import Tensor
from transformer_lens import HookedTransformer, ActivationCache


# ---------------------------------------------------------------------------
# Activation function derivatives
# ---------------------------------------------------------------------------

def _gelu_derivative(x: Tensor) -> Tensor:
    """Derivative of GeLU (tanh approximation)."""
    cdf = 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))
    inner = math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))
    inner_d = math.sqrt(2 / math.pi) * (1 + 0.134145 * x.pow(2))
    pdf = 0.5 * inner_d * (1 - torch.tanh(inner).pow(2))
    return cdf + x * pdf


def _silu_derivative(x: Tensor) -> Tensor:
    """Derivative of SiLU (Swish)."""
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


def _act_derivative(x: Tensor, act_fn: str) -> Tensor:
    if 'silu' in act_fn:
        return _silu_derivative(x)
    return _gelu_derivative(x)


def _act_forward(x: Tensor, act_fn: str) -> Tensor:
    if 'silu' in act_fn:
        return torch.nn.functional.silu(x)
    return torch.nn.functional.gelu(x)


# ---------------------------------------------------------------------------
# Rotary embedding helpers
# ---------------------------------------------------------------------------

def _rotate_half(x: Tensor, adjacent_pairs: bool = False) -> Tensor:
    rot = x.clone()
    if adjacent_pairs:
        rot[..., ::2] = -x[..., 1::2]
        rot[..., 1::2] = x[..., ::2]
    else:
        n = x.size(-1) // 2
        rot[..., :n] = -x[..., n:]
        rot[..., n:] = x[..., :n]
    return rot


def _apply_rotary_tangent(
    delta: Tensor,
    block,
    seq_len: int,
    cfg,
) -> Tensor:
    """Apply rotary embeddings to tangent Q/K vectors.

    delta: (batch, chunk, pos, n_heads, d_head)
    Returns: same shape with rotary applied to first rotary_dim dimensions.
    """
    rotary_dim = cfg.rotary_dim or cfg.d_head
    adjacent = getattr(cfg, 'rotary_adjacent_pairs', False)
    sin = block.attn.rotary_sin[:seq_len]
    cos = block.attn.rotary_cos[:seq_len]

    shape = delta.shape
    flat = delta.reshape(-1, *shape[-3:])

    x_rot = flat[..., :rotary_dim]
    x_pass = flat[..., rotary_dim:]

    cos = cos.unsqueeze(0).unsqueeze(-2)
    sin = sin.unsqueeze(0).unsqueeze(-2)

    x_rotated = x_rot * cos + _rotate_half(x_rot, adjacent) * sin
    result = torch.cat([x_rotated, x_pass], dim=-1)
    return result.reshape(shape)


# ---------------------------------------------------------------------------
# JVP through individual blocks
# ---------------------------------------------------------------------------

def jvp_attention(
    tangent: Tensor,
    layer_idx: int,
    model: HookedTransformer,
    cache: ActivationCache,
    tsg_temperature: float,
    scale_multiplicative: bool,
) -> Tensor:
    """GIM-corrected JVP through the attention block (frozen LN1 + attention + residual).

    tangent: (batch, chunk, pos, d_model)
    Returns: (batch, chunk, pos, d_model) = tangent + delta_attn_result
    """
    cfg = model.cfg
    block = model.blocks[layer_idx]
    pos = tangent.shape[2]

    # --- Frozen LN1 ---
    ln1_scale = cache[f'blocks.{layer_idx}.ln1.hook_scale']
    if ln1_scale.ndim == 4:
        ln1_scale = ln1_scale[:, :, 0, :]  # collapse head dim
    t_ln = tangent / ln1_scale.unsqueeze(1)
    if hasattr(block.ln1, 'w'):
        t_ln = t_ln * block.ln1.w

    # --- Q, K, V projections ---
    W_Q = model.W_Q[layer_idx]
    W_K = model.W_K[layer_idx]
    W_V = model.W_V[layer_idx]

    dq = torch.einsum('bcpd,ndh->bcpnh', t_ln, W_Q)
    dk = torch.einsum('bcpd,ndh->bcpnh', t_ln, W_K)
    dv = torch.einsum('bcpd,ndh->bcpnh', t_ln, W_V)

    # --- Rotary embeddings ---
    if cfg.positional_embedding_type == "rotary":
        dq = _apply_rotary_tangent(dq, block, pos, cfg)
        dk = _apply_rotary_tangent(dk, block, pos, cfg)
        Q_cached = cache[f'blocks.{layer_idx}.attn.hook_rot_q']
        K_cached = cache[f'blocks.{layer_idx}.attn.hook_rot_k']
    else:
        Q_cached = cache[f'blocks.{layer_idx}.attn.hook_q']
        K_cached = cache[f'blocks.{layer_idx}.attn.hook_k']

    V_cached = cache[f'blocks.{layer_idx}.attn.hook_v']
    attn_scale = math.sqrt(cfg.d_head)

    # --- Attention score JVP ---
    d_scores = (
        torch.einsum('bcqnh,bknh->bcnqk', dq, K_cached) +
        torch.einsum('bqnh,bcknh->bcnqk', Q_cached, dk)
    ) / attn_scale

    if scale_multiplicative:
        d_scores = d_scores / 2  # Shapley at Q*K

    # --- Soft cap (Gemma 2 etc.) ---
    soft_cap = getattr(cfg, 'attn_scores_soft_cap', 0.0)
    if soft_cap:
        raw_scores = cache[f'blocks.{layer_idx}.attn.hook_attn_scores']
        cap_deriv = 1 - torch.tanh(raw_scores / soft_cap).pow(2)
        d_scores = d_scores * cap_deriv.unsqueeze(1)

    # --- TSG softmax JVP ---
    attn_scores = cache[f'blocks.{layer_idx}.attn.hook_attn_scores']
    alpha_tsg = torch.softmax(attn_scores / tsg_temperature, dim=-1)
    a = alpha_tsg.unsqueeze(1)
    dot = (a * d_scores).sum(-1, keepdim=True)
    d_alpha = a * (d_scores - dot)

    # --- Attention output JVP ---
    alpha_cached = cache[f'blocks.{layer_idx}.attn.hook_pattern']
    d_attn_out = (
        torch.einsum('bcnqk,bknh->bcqnh', d_alpha, V_cached) +
        torch.einsum('bnqk,bcknh->bcqnh', alpha_cached, dv)
    )

    if scale_multiplicative:
        d_attn_out = d_attn_out / 2  # Shapley at alpha*V

    # --- W_O projection (sum over heads) ---
    W_O = model.W_O[layer_idx]
    d_result = torch.einsum('bcqnh,nhd->bcqd', d_attn_out, W_O)

    # --- Residual ---
    return tangent + d_result


def jvp_mlp(
    tangent: Tensor,
    layer_idx: int,
    model: HookedTransformer,
    cache: ActivationCache,
    scale_multiplicative: bool,
) -> Tensor:
    """GIM-corrected JVP through the MLP block (frozen LN2 + MLP + residual).

    tangent: (batch, chunk, pos, d_model)
    Returns: (batch, chunk, pos, d_model) = tangent + delta_mlp_out
    """
    cfg = model.cfg
    block = model.blocks[layer_idx]

    # --- Frozen LN2 ---
    ln2_scale = cache[f'blocks.{layer_idx}.ln2.hook_scale']
    t_ln = tangent / ln2_scale.unsqueeze(1)
    if hasattr(block.ln2, 'w'):
        t_ln = t_ln * block.ln2.w

    if cfg.gated_mlp:
        W_in = block.mlp.W_in
        W_gate = block.mlp.W_gate
        W_out = block.mlp.W_out

        gate_pre = cache[f'blocks.{layer_idx}.mlp.hook_pre']
        in_linear = cache[f'blocks.{layer_idx}.mlp.hook_pre_linear']

        act_deriv = _act_derivative(gate_pre, cfg.act_fn)
        act_val = _act_forward(gate_pre, cfg.act_fn)

        d_gate = torch.einsum('bcpd,dm->bcpm', t_ln, W_gate)
        d_in = torch.einsum('bcpd,dm->bcpm', t_ln, W_in)

        d_gate_act = act_deriv.unsqueeze(1) * d_gate
        d_mid = d_gate_act * in_linear.unsqueeze(1) + act_val.unsqueeze(1) * d_in

        if scale_multiplicative:
            d_mid = d_mid / 2

        d_mlp_out = torch.einsum('bcpm,md->bcpd', d_mid, W_out)
    else:
        W_in = block.mlp.W_in
        W_out = block.mlp.W_out

        pre = cache[f'blocks.{layer_idx}.mlp.hook_pre']
        act_deriv = _act_derivative(pre, cfg.act_fn)

        d_pre = torch.einsum('bcpd,dm->bcpm', t_ln, W_in)
        d_post = act_deriv.unsqueeze(1) * d_pre
        d_mlp_out = torch.einsum('bcpm,md->bcpd', d_post, W_out)

    return tangent + d_mlp_out


# ---------------------------------------------------------------------------
# Chunked JVP helpers (propagate all sources through one half-block)
# ---------------------------------------------------------------------------

def _propagate_chunked_attention(
    source_acts: Tensor,
    n_src: int,
    layer_idx: int,
    model: HookedTransformer,
    cache: ActivationCache,
    tsg_temperature: float,
    scale_multiplicative: bool,
    chunk_size: int,
) -> Tensor:
    """Propagate source_acts[:, :, :n_src] through attention of layer_idx in chunks.

    Modifies and returns source_acts with the first n_src sources propagated.
    """
    batch, pos, _, d_model = source_acts.shape
    results = []
    for start in range(0, n_src, chunk_size):
        end = min(start + chunk_size, n_src)
        tangent = source_acts[:, :, start:end].permute(0, 2, 1, 3)  # (batch, chunk, pos, d)
        tangent = jvp_attention(tangent, layer_idx, model, cache, tsg_temperature, scale_multiplicative)
        results.append(tangent.permute(0, 2, 1, 3))
    source_acts = source_acts.clone()
    source_acts[:, :, :n_src] = torch.cat(results, dim=2)
    return source_acts


def _propagate_chunked_mlp(
    source_acts: Tensor,
    n_src: int,
    layer_idx: int,
    model: HookedTransformer,
    cache: ActivationCache,
    scale_multiplicative: bool,
    chunk_size: int,
) -> Tensor:
    """Propagate source_acts[:, :, :n_src] through MLP of layer_idx in chunks."""
    batch, pos, _, d_model = source_acts.shape
    results = []
    for start in range(0, n_src, chunk_size):
        end = min(start + chunk_size, n_src)
        tangent = source_acts[:, :, start:end].permute(0, 2, 1, 3)
        tangent = jvp_mlp(tangent, layer_idx, model, cache, scale_multiplicative)
        results.append(tangent.permute(0, 2, 1, 3))
    source_acts = source_acts.clone()
    source_acts[:, :, :n_src] = torch.cat(results, dim=2)
    return source_acts


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _proximity_per_sample(
    contributions: Tensor,
    reference: Tensor,
) -> Tensor:
    """Compute normalized ALTI proximity per sample and position (no aggregation).

    contributions: (batch, pos, n_src, d_model)
    reference: (batch, pos, d_model)

    Returns: (batch, pos, n_src) normalized proximity weights (sum to 1 across sources).
    """
    ref_unsq = reference.unsqueeze(2)
    dist = torch.linalg.vector_norm(contributions - ref_unsq, ord=1, dim=-1)
    ref_norm = torch.linalg.vector_norm(ref_unsq, ord=1, dim=-1)
    proximity = torch.clamp(-dist + ref_norm, min=0)
    prox_sum = proximity.sum(dim=2, keepdim=True).clamp(min=1e-10)
    return proximity / prox_sum


def _gradient_projection_per_sample(
    contributions: Tensor,
    grad: Tensor,
) -> Tensor:
    """Compute per-sample gradient projection (no aggregation).

    contributions: (batch, pos, n_src, d_model)
    grad: (batch, pos, d_model) or (batch, pos, n_heads, d_model)

    Returns: (batch, pos, n_src) or (batch, pos, n_src, n_heads)
    """
    if grad.ndim == 4:
        return torch.einsum('bpsd,bphd->bpsh', contributions, grad)
    return (contributions * grad.unsqueeze(2)).sum(dim=-1)


def _mask_and_aggregate(
    scores: Tensor,
    input_lengths: Tensor,
) -> Tensor:
    """Mask padding positions and aggregate over positions and batch.

    scores: (batch, pos, n_src, ...) — may have trailing n_heads dim
    input_lengths: (batch,)

    Returns: (n_src,) or (n_src, n_heads)
    """
    max_len = input_lengths.max()
    mask = torch.arange(max_len, device=input_lengths.device, dtype=input_lengths.dtype
                        ).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
    # expand mask to match scores dims
    for _ in range(scores.ndim - 2):
        mask = mask.unsqueeze(-1)
    scores = scores * mask
    scores = scores.sum(dim=1)  # sum over positions
    return scores.sum(dim=0)    # sum over batch


def compute_edge_gradient_scores(
    contributions: Tensor,
    grad: Tensor,
    input_lengths: Tensor,
) -> Tensor:
    """Score edges via dot product with destination gradient.

    score_j = Σ_batch Σ_pos z_j[d] × grad_dest[d]

    contributions: (batch, pos, n_src, d_model)
    grad: (batch, pos, d_model) or (batch, pos, n_heads, d_model)
    input_lengths: (batch,)

    Returns: (n_src,) or (n_src, n_heads)
    """
    scores = _gradient_projection_per_sample(contributions, grad)
    return _mask_and_aggregate(scores, input_lengths)


def compute_combined_scores(
    contributions: Tensor,
    reference: Tensor,
    grad: Tensor,
    input_lengths: Tensor,
) -> Tensor:
    """Score edges by proximity-weighted gradient projection (GWAI).

    Per position and sample:
      score_j = proximity(z_j, y) × (z_j · grad_dest)

    ALTI proximity provides structural weights (how much does this source
    account for the representation), gradient projection provides task
    direction (in what direction should the representation change).
    The product selects edges that are both structurally important AND
    task-relevant.

    contributions: (batch, pos, n_src, d_model)
    reference: (batch, pos, d_model)
    grad: (batch, pos, d_model) or (batch, pos, n_heads, d_model)
    input_lengths: (batch,)

    Returns: (n_src,) or (n_src, n_heads)
    """
    proximity = _proximity_per_sample(contributions, reference)  # (batch, pos, n_src)
    grad_proj = _gradient_projection_per_sample(contributions, grad)

    per_head = grad.ndim == 4
    if per_head:
        # proximity: (batch, pos, n_src) -> (batch, pos, n_src, 1)
        # grad_proj: (batch, pos, n_src, n_heads)
        combined = proximity.unsqueeze(-1) * grad_proj
    else:
        combined = proximity * grad_proj  # (batch, pos, n_src)

    return _mask_and_aggregate(combined, input_lengths)


def compute_proximity_scores(
    contributions: Tensor,
    reference: Tensor,
    input_lengths: Tensor,
) -> Tensor:
    """Compute ALTI proximity-based importance scores.

    contributions: (batch, pos, n_src, d_model)
    reference: (batch, pos, d_model)
    input_lengths: (batch,)

    Returns: (n_src,) importance scores aggregated over batch and positions.
    """
    ref_unsq = reference.unsqueeze(2)
    dist = torch.linalg.vector_norm(contributions - ref_unsq, ord=1, dim=-1)
    ref_norm = torch.linalg.vector_norm(ref_unsq, ord=1, dim=-1)
    proximity = torch.clamp(-dist + ref_norm, min=0)

    prox_sum = proximity.sum(dim=2, keepdim=True).clamp(min=1e-10)
    importance = proximity / prox_sum

    max_len = input_lengths.max()
    mask = torch.arange(max_len, device=input_lengths.device, dtype=input_lengths.dtype
                        ).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)
    importance = importance * mask

    importance = importance.sum(dim=1) / input_lengths.view(-1, 1)
    return importance.sum(dim=0)


# ---------------------------------------------------------------------------
# Cache filter
# ---------------------------------------------------------------------------

def make_names_filter(model: HookedTransformer, needs_jvp: bool):
    """Create a names_filter for run_with_cache to only cache needed activations."""
    needed = {'hook_embed'}
    n_layers = model.cfg.n_layers
    rotary = model.cfg.positional_embedding_type == "rotary"

    for l in range(n_layers):
        needed.add(f'blocks.{l}.attn.hook_result')
        needed.add(f'blocks.{l}.hook_mlp_out')
        needed.add(f'blocks.{l}.hook_resid_pre')
        if not model.cfg.parallel_attn_mlp:
            needed.add(f'blocks.{l}.hook_resid_mid')
        needed.add(f'blocks.{l}.hook_resid_post')

    if needs_jvp:
        for l in range(n_layers):
            needed.add(f'blocks.{l}.ln1.hook_scale')
            needed.add(f'blocks.{l}.ln2.hook_scale')
            needed.add(f'blocks.{l}.attn.hook_attn_scores')
            needed.add(f'blocks.{l}.attn.hook_pattern')
            needed.add(f'blocks.{l}.attn.hook_v')
            if rotary:
                needed.add(f'blocks.{l}.attn.hook_rot_q')
                needed.add(f'blocks.{l}.attn.hook_rot_k')
            else:
                needed.add(f'blocks.{l}.attn.hook_q')
                needed.add(f'blocks.{l}.attn.hook_k')
            needed.add(f'blocks.{l}.mlp.hook_pre')
            if model.cfg.gated_mlp:
                needed.add(f'blocks.{l}.mlp.hook_pre_linear')

    return lambda name: name in needed

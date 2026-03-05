"""GWAI: GIM-corrected ALTI with downstream look-ahead for circuit discovery.

Implements the hybrid ALTI-GIM approach that propagates edge contributions
through k subsequent layers using GIM-corrected Jacobian-vector products (JVPs),
then scores edges using the ALTI proximity metric at the downstream point.

k=0 recovers pure ALTI (information-flow-routes). k>=1 accounts for self-repair
effects via temperature-adjusted softmax gradients (TSG), frozen LayerNorm, and
Shapley-based gradient normalization at multiplicative interactions.
"""

import math
from typing import Optional

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
    sin = block.attn.rotary_sin[:seq_len]  # (pos, rotary_dim)
    cos = block.attn.rotary_cos[:seq_len]  # (pos, rotary_dim)

    # Flatten leading dims so we have (flat, pos, n_heads, d_head)
    shape = delta.shape
    flat = delta.reshape(-1, *shape[-3:])

    x_rot = flat[..., :rotary_dim]
    x_pass = flat[..., rotary_dim:]

    # Broadcast: (1, pos, 1, rotary_dim)
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
    # With use_split_qkv_input, scale is (batch, pos, n_heads, 1); collapse to (batch, pos, 1)
    if ln1_scale.ndim == 4:
        ln1_scale = ln1_scale[:, :, 0, :]  # identical across heads
    t_ln = tangent / ln1_scale.unsqueeze(1)  # (batch, 1, pos, 1)
    if hasattr(block.ln1, 'w'):
        t_ln = t_ln * block.ln1.w  # (d_model,)

    # --- Q, K, V projections ---
    W_Q = model.W_Q[layer_idx]  # (n_heads, d_model, d_head)
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

    V_cached = cache[f'blocks.{layer_idx}.attn.hook_v']  # (batch, pos, n_heads, d_head)
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
    attn_scores = cache[f'blocks.{layer_idx}.attn.hook_attn_scores']  # (batch, n_heads, q, k)
    alpha_tsg = torch.softmax(attn_scores / tsg_temperature, dim=-1)
    a = alpha_tsg.unsqueeze(1)  # (batch, 1, n_heads, q, k)
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
    W_O = model.W_O[layer_idx]  # (n_heads, d_head, d_model)
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
    ln2_scale = cache[f'blocks.{layer_idx}.ln2.hook_scale']  # (batch, pos, 1)
    t_ln = tangent / ln2_scale.unsqueeze(1)
    if hasattr(block.ln2, 'w'):
        t_ln = t_ln * block.ln2.w

    if cfg.gated_mlp:
        W_in = block.mlp.W_in      # (d_model, d_mlp)
        W_gate = block.mlp.W_gate  # (d_model, d_mlp)
        W_out = block.mlp.W_out    # (d_mlp, d_model)

        gate_pre = cache[f'blocks.{layer_idx}.mlp.hook_pre']         # (batch, pos, d_mlp)
        in_linear = cache[f'blocks.{layer_idx}.mlp.hook_pre_linear'] # (batch, pos, d_mlp)

        act_deriv = _act_derivative(gate_pre, cfg.act_fn)
        act_val = _act_forward(gate_pre, cfg.act_fn)

        d_gate = torch.einsum('bcpd,dm->bcpm', t_ln, W_gate)
        d_in = torch.einsum('bcpd,dm->bcpm', t_ln, W_in)

        d_gate_act = act_deriv.unsqueeze(1) * d_gate
        d_mid = d_gate_act * in_linear.unsqueeze(1) + act_val.unsqueeze(1) * d_in

        if scale_multiplicative:
            d_mid = d_mid / 2  # Shapley at gate*input

        d_mlp_out = torch.einsum('bcpm,md->bcpd', d_mid, W_out)
    else:
        W_in = block.mlp.W_in    # (d_model, d_mlp)
        W_out = block.mlp.W_out  # (d_mlp, d_model)

        pre = cache[f'blocks.{layer_idx}.mlp.hook_pre']  # (batch, pos, d_mlp)
        act_deriv = _act_derivative(pre, cfg.act_fn)

        d_pre = torch.einsum('bcpd,dm->bcpm', t_ln, W_in)
        d_post = act_deriv.unsqueeze(1) * d_pre
        d_mlp_out = torch.einsum('bcpm,md->bcpd', d_post, W_out)

    return tangent + d_mlp_out


def jvp_layer(
    tangent: Tensor,
    layer_idx: int,
    model: HookedTransformer,
    cache: ActivationCache,
    tsg_temperature: float,
    scale_multiplicative: bool,
) -> Tensor:
    """GIM-corrected JVP through a full transformer block (attention + MLP).

    For non-parallel architectures: attention first, then MLP.
    For parallel architectures: attention and MLP in parallel.
    """
    if model.cfg.parallel_attn_mlp:
        d_attn = jvp_attention(tangent, layer_idx, model, cache, tsg_temperature, scale_multiplicative) - tangent
        d_mlp = jvp_mlp(tangent, layer_idx, model, cache, scale_multiplicative) - tangent
        return tangent + d_attn + d_mlp
    else:
        mid = jvp_attention(tangent, layer_idx, model, cache, tsg_temperature, scale_multiplicative)
        return jvp_mlp(mid, layer_idx, model, cache, scale_multiplicative)


# ---------------------------------------------------------------------------
# Multi-layer propagation
# ---------------------------------------------------------------------------

def propagate(
    contributions: Tensor,
    dest_layer: int,
    dest_type: str,
    k: int,
    model: HookedTransformer,
    cache: ActivationCache,
    tsg_temperature: float,
    scale_multiplicative: bool,
    chunk_size: int,
) -> Tensor:
    """Propagate source contributions through k layers with GIM corrections.

    contributions: (batch, pos, n_src, d_model)
    dest_layer: the layer of the destination node
    dest_type: 'attention', 'mlp', or 'logits'
    k: number of look-ahead layers (0 = pure ALTI)

    Returns: (batch, pos, n_src, d_model) propagated contributions.
    """
    if k == 0:
        return contributions

    n_layers = model.cfg.n_layers
    batch, pos, n_src, d_model = contributions.shape

    # Determine which layers to propagate through.
    # For attention destinations: propagate through the full block at dest_layer, then k-1 more.
    # For MLP destinations (non-parallel): propagate through MLP of dest_layer, then k-1 more.
    # For logits: no layers after, so return as-is.
    if dest_type == 'logits':
        return contributions

    # Process in chunks to manage memory
    results = []
    for start in range(0, n_src, chunk_size):
        end = min(start + chunk_size, n_src)
        # tangent: (batch, chunk, pos, d_model)
        tangent = contributions[:, :, start:end].permute(0, 2, 1, 3)

        if dest_type == 'attention':
            # Propagate through full blocks: dest_layer, dest_layer+1, ..., dest_layer+k-1
            for l in range(dest_layer, min(dest_layer + k, n_layers)):
                tangent = jvp_layer(tangent, l, model, cache, tsg_temperature, scale_multiplicative)
        elif dest_type == 'mlp':
            # First: propagate through MLP of dest_layer (half-block)
            if dest_layer < n_layers:
                tangent = jvp_mlp(tangent, dest_layer, model, cache, scale_multiplicative)
            # Then: propagate through full blocks dest_layer+1 to dest_layer+k-1
            for l in range(dest_layer + 1, min(dest_layer + k, n_layers)):
                tangent = jvp_layer(tangent, l, model, cache, tsg_temperature, scale_multiplicative)

        # tangent: (batch, chunk, pos, d_model) -> (batch, pos, chunk, d_model)
        results.append(tangent.permute(0, 2, 1, 3))

    return torch.cat(results, dim=2)


# ---------------------------------------------------------------------------
# Reference point computation
# ---------------------------------------------------------------------------

def get_reference(
    dest_layer: int,
    dest_type: str,
    k: int,
    model: HookedTransformer,
    cache: ActivationCache,
) -> Tensor:
    """Get the reference residual stream for proximity comparison.

    For k=0: the residual stream at the destination's input point (ALTI behavior).
    For k>=1: the residual stream k layers downstream from the destination.

    Returns: (batch, pos, d_model)
    """
    n_layers = model.cfg.n_layers

    if k == 0:
        if dest_type == 'attention':
            return cache[f'blocks.{dest_layer}.hook_resid_pre']
        elif dest_type == 'mlp':
            if model.cfg.parallel_attn_mlp:
                return cache[f'blocks.{dest_layer}.hook_resid_pre']
            else:
                return cache[f'blocks.{dest_layer}.hook_resid_mid']
        else:  # logits
            return cache[f'blocks.{n_layers - 1}.hook_resid_post']
    else:
        if dest_type == 'logits':
            return cache[f'blocks.{n_layers - 1}.hook_resid_post']

        if dest_type == 'attention':
            ref_layer = min(dest_layer + k - 1, n_layers - 1)
        else:  # mlp
            ref_layer = min(dest_layer + k - 1, n_layers - 1)

        return cache[f'blocks.{ref_layer}.hook_resid_post']


# ---------------------------------------------------------------------------
# ALTI proximity scoring
# ---------------------------------------------------------------------------

def compute_proximity_scores(
    contributions: Tensor,
    reference: Tensor,
    input_lengths: Tensor,
    renormalize: bool,
) -> Tensor:
    """Compute ALTI proximity-based importance scores.

    contributions: (batch, pos, n_src, d_model)
    reference: (batch, pos, d_model)
    input_lengths: (batch,)
    renormalize: if True, use sum of contributions as reference instead of
                 actual residual stream (recommended for k>0).

    Returns: (n_src,) importance scores aggregated over batch and positions.
    """
    if renormalize:
        ref = contributions.sum(dim=2)  # (batch, pos, d_model)
    else:
        ref = reference

    # proximity(z, y) = max(-||z - y||_1 + ||y||_1, 0)
    # contributions: (batch, pos, n_src, d_model)
    # ref: (batch, pos, d_model) -> (batch, pos, 1, d_model)
    ref_unsq = ref.unsqueeze(2)
    dist = torch.linalg.vector_norm(contributions - ref_unsq, ord=1, dim=-1)
    ref_norm = torch.linalg.vector_norm(ref_unsq, ord=1, dim=-1)
    proximity = torch.clamp(-dist + ref_norm, min=0)
    # proximity: (batch, pos, n_src)

    # Normalize over source dimension
    prox_sum = proximity.sum(dim=2, keepdim=True)
    # Avoid division by zero
    prox_sum = prox_sum.clamp(min=1e-10)
    importance = proximity / prox_sum
    # importance: (batch, pos, n_src)

    # Mask padding positions
    max_len = input_lengths.max()
    mask = torch.arange(max_len, device=input_lengths.device, dtype=input_lengths.dtype
                        ).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1)  # (batch, pos, 1)
    importance = importance * mask

    # Mean over positions, sum over batch
    importance = importance.sum(dim=1) / input_lengths.view(-1, 1)  # (batch, n_src)
    importance = importance.sum(dim=0)  # (n_src,)

    return importance


# ---------------------------------------------------------------------------
# Cache filter
# ---------------------------------------------------------------------------

def make_names_filter(model: HookedTransformer, k: int):
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

    if k >= 1:
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

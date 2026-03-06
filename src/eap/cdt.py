"""CD-T: Contextual Decomposition for Transformers (Hsu et al., 2024).

Decomposes activations into relevant (rel) and irrelevant (irrel) components
to compute edge-level importance scores for circuit discovery.

Two modes:
- ``full=False`` (default): Source output is extracted via hooks and set as
  ``rel``; CD propagation handles subsequent layers.
- ``full=True``: Full CD-T decomposition from embeddings.  The source head is
  marked in the attention value space (after softmax·V, before W_O) following
  the original paper, so the score reflects how much signal the attention
  mechanism *routes through* that head.

Reference: https://arxiv.org/abs/2407.00886
"""

import math
from typing import Tuple

import torch
from torch import Tensor, nn
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Core decomposition primitives
# ---------------------------------------------------------------------------

def _normalize(rel: Tensor, irrel: Tensor) -> None:
    """In-place stabilisation: when rel/irrel have opposite signs at the same
    element, consolidate into whichever has larger magnitude."""
    tot = rel + irrel
    conflict = (rel * irrel) < 0
    rel_wins = conflict & (rel.abs() >= irrel.abs())
    irrel_wins = conflict & (~rel_wins)
    rel[rel_wins] = tot[rel_wins]
    rel[irrel_wins] = 0
    irrel[irrel_wins] = tot[irrel_wins]
    irrel[rel_wins] = 0


def cd_linear(rel: Tensor, irrel: Tensor, W: Tensor, b: Tensor,
              tol: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """x @ W + b  with proportional bias split."""
    r = rel @ W
    ir = irrel @ W
    denom = r.abs() + ir.abs() + tol
    frac = r.abs() / denom
    b_exp = b.expand_as(r)
    return r + b_exp * frac, ir + b_exp * (1 - frac)


def cd_layer_norm(rel: Tensor, irrel: Tensor,
                  w: Tensor, b: Tensor, eps: float = 1e-5,
                  tol: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """LayerNorm decomposition (pre-norm GPT style)."""
    tot = rel + irrel
    r_mean = rel.mean(dim=-1, keepdim=True)
    ir_mean = irrel.mean(dim=-1, keepdim=True)
    var = tot.pow(2).mean(-1, keepdim=True) - tot.mean(-1, keepdim=True).pow(2)
    inv_std = (var + eps).rsqrt()

    r_out = (rel - r_mean) * inv_std * w
    ir_out = (irrel - ir_mean) * inv_std * w

    frac = rel.abs() / (rel.abs() + irrel.abs() + tol)
    return r_out + b * frac, ir_out + b * (1 - frac)


def cd_gelu(rel: Tensor, irrel: Tensor) -> Tuple[Tensor, Tensor]:
    """GELU decomposition (ACD rule)."""
    ir_act = nn.functional.gelu(irrel)
    return nn.functional.gelu(rel + irrel) - ir_act, ir_act


def _get_eps(ln_module) -> float:
    if hasattr(ln_module, 'cfg') and hasattr(ln_module.cfg, 'eps'):
        return ln_module.cfg.eps
    return 1e-5


# ---------------------------------------------------------------------------
# Attention decomposition
# ---------------------------------------------------------------------------

def cd_attention(rel: Tensor, irrel: Tensor, block,
                 causal_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Decompose self-attention (no LN, no residual).
    Returns summed (batch, pos, d_model) output."""
    attn = block.attn
    tol = 1e-8
    scale = math.sqrt(attn.cfg.d_head)

    def _proj(r, ir, W, b):
        rp = torch.einsum("bpm,nmh->bpnh", r, W)
        ip = torch.einsum("bpm,nmh->bpnh", ir, W)
        denom = rp.abs() + ip.abs() + tol
        f = rp.abs() / denom
        be = b.expand_as(rp)
        return rp + be * f, ip + be * (1 - f)

    rq, iq = _proj(rel, irrel, attn.W_Q, attn.b_Q)
    rk, ik = _proj(rel, irrel, attn.W_K, attn.b_K)
    rv, iv = _proj(rel, irrel, attn.W_V, attn.b_V)

    tq, tk = rq + iq, rk + ik
    tot_scores = torch.einsum("bqnh,bknh->bnqk", tq, tk) / scale + causal_mask
    rel_scores = torch.einsum("bqnh,bknh->bnqk", rq, rk) / scale + causal_mask
    tot_probs = nn.functional.softmax(tot_scores, dim=-1)
    rel_probs = nn.functional.softmax(rel_scores, dim=-1)

    tv = rv + iv
    tot_ctx = torch.einsum("bnqk,bknh->bqnh", tot_probs, tv)
    rel_ctx = torch.einsum("bnqk,bknh->bqnh", rel_probs, rv)
    irrel_ctx = tot_ctx - rel_ctx

    r_out = torch.einsum("bqnh,nhm->bqm", rel_ctx, attn.W_O)
    ir_out = torch.einsum("bqnh,nhm->bqm", irrel_ctx, attn.W_O)
    denom = r_out.abs() + ir_out.abs() + tol
    f = r_out.abs() / denom
    be = attn.b_O.expand_as(r_out)
    return r_out + be * f, ir_out + be * (1 - f)


def cd_attention_with_mark(
    rel: Tensor, irrel: Tensor, block, causal_mask: Tensor,
    mark_head: int = -1,
) -> Tuple[Tensor, Tensor]:
    """Decompose self-attention with optional source marking in value space.

    If ``mark_head >= 0``, after computing softmax·V (in d_head space per
    head), the specified head's irrel is moved to rel — matching the original
    CD-T paper's ``set_rel_at_source_nodes`` which operates between the
    attention mechanism and the W_O projection.

    Returns: (rel_out, irrel_out) each (batch, pos, d_model).
    """
    attn = block.attn
    tol = 1e-8
    scale = math.sqrt(attn.cfg.d_head)

    def _proj(r, ir, W, b):
        rp = torch.einsum("bpm,nmh->bpnh", r, W)
        ip = torch.einsum("bpm,nmh->bpnh", ir, W)
        denom = rp.abs() + ip.abs() + tol
        f = rp.abs() / denom
        be = b.expand_as(rp)
        return rp + be * f, ip + be * (1 - f)

    rq, iq = _proj(rel, irrel, attn.W_Q, attn.b_Q)
    rk, ik = _proj(rel, irrel, attn.W_K, attn.b_K)
    rv, iv = _proj(rel, irrel, attn.W_V, attn.b_V)

    tq, tk = rq + iq, rk + ik
    tot_scores = torch.einsum("bqnh,bknh->bnqk", tq, tk) / scale + causal_mask
    rel_scores = torch.einsum("bqnh,bknh->bnqk", rq, rk) / scale + causal_mask
    tot_probs = nn.functional.softmax(tot_scores, dim=-1)
    rel_probs = nn.functional.softmax(rel_scores, dim=-1)

    tv = rv + iv
    tot_ctx = torch.einsum("bnqk,bknh->bqnh", tot_probs, tv)
    rel_ctx = torch.einsum("bnqk,bknh->bqnh", rel_probs, rv)
    irrel_ctx = tot_ctx - rel_ctx

    # --- Source marking in value space (before W_O) ---
    if mark_head >= 0:
        h = mark_head
        # Move irrel -> rel for the marked head at all positions
        rel_ctx[:, :, h, :] = rel_ctx[:, :, h, :] + irrel_ctx[:, :, h, :]
        irrel_ctx[:, :, h, :] = 0

    # W_O projection
    r_out = torch.einsum("bqnh,nhm->bqm", rel_ctx, attn.W_O)
    ir_out = torch.einsum("bqnh,nhm->bqm", irrel_ctx, attn.W_O)
    denom = r_out.abs() + ir_out.abs() + tol
    f = r_out.abs() / denom
    be = attn.b_O.expand_as(r_out)
    return r_out + be * f, ir_out + be * (1 - f)


# ---------------------------------------------------------------------------
# MLP decomposition
# ---------------------------------------------------------------------------

def cd_mlp(rel: Tensor, irrel: Tensor, block) -> Tuple[Tensor, Tensor]:
    """Decompose MLP sub-layer (expects post-LN2 input, no residual)."""
    mlp = block.mlp
    r, ir = cd_linear(rel, irrel, mlp.W_in, mlp.b_in)
    _normalize(r, ir)
    r, ir = cd_gelu(r, ir)
    r, ir = cd_linear(r, ir, mlp.W_out, mlp.b_out)
    _normalize(r, ir)
    return r, ir


# ---------------------------------------------------------------------------
# Edge-level scoring helpers
# ---------------------------------------------------------------------------

def _l1(x: Tensor, mask: Tensor) -> float:
    """Sum of L1 norms over batch & masked positions.
    x: (batch, pos, ...), mask: (batch, pos)."""
    return (x.abs().sum(dim=-1) * mask).sum().item()


def _build_causal_mask(n_pos: int, attention_mask, device, dtype) -> Tensor:
    """Causal + padding mask: (1, 1, n_pos, n_pos)."""
    causal = torch.triu(
        torch.full((n_pos, n_pos), float('-inf'), device=device, dtype=dtype),
        diagonal=1).unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        pad_mask = (1 - attention_mask).bool().unsqueeze(1).unsqueeze(2)
        causal = causal.masked_fill(pad_mask, float('-inf'))
    return causal


# ---------------------------------------------------------------------------
# Hook-based scoring (full=False, the default)
# ---------------------------------------------------------------------------

def cd_edge_scores(
    model: HookedTransformer,
    tokens: Tensor,
    attention_mask: Tensor,
    input_lengths: Tensor,
    n_forward: int,
    n_backward: int,
    n_layers: int,
    n_heads: int,
) -> Tensor:
    """Edge scores using hook-based source extraction + CD propagation."""
    device = tokens.device
    dtype = model.cfg.dtype or torch.float32
    batch, n_pos = tokens.shape

    causal = _build_causal_mask(n_pos, attention_mask, device, dtype)
    pos_mask = (torch.arange(n_pos, device=device).unsqueeze(0)
                < input_lengths.unsqueeze(1)).to(dtype)

    scores = torch.zeros(n_forward, n_backward, device=device, dtype=torch.float32)

    # Phase 1: cache via hooks
    from functools import partial
    resid_pre, head_outputs, mlp_outputs = [], [], []

    def _cap_r(l, a, hook): resid_pre.append(a.detach())
    def _cap_h(l, a, hook): head_outputs.append(a.detach())
    def _cap_m(l, a, hook): mlp_outputs.append(a.detach())

    fwd_hooks = []
    for l in range(n_layers):
        fwd_hooks.append((f"blocks.{l}.hook_resid_pre", partial(_cap_r, l)))
        fwd_hooks.append((f"blocks.{l}.attn.hook_result", partial(_cap_h, l)))
        fwd_hooks.append((f"blocks.{l}.hook_mlp_out", partial(_cap_m, l)))

    with torch.inference_mode():
        embeddings = model.embed(tokens) + model.pos_embed(tokens)
        with model.hooks(fwd_hooks=fwd_hooks):
            model(tokens, attention_mask=attention_mask)

    resid_mid = []
    for l in range(n_layers):
        resid_mid.append(resid_pre[l] + head_outputs[l].sum(dim=2) + model.blocks[l].attn.b_O)

    # Phase 2: propagate
    def _score_qkv(rel_ln, irrel_ln, layer, src_fwd):
        attn = model.blocks[layer].attn
        tol = 1e-8
        for qkv_off, (W, b) in enumerate([
            (attn.W_Q, attn.b_Q), (attn.W_K, attn.b_K), (attn.W_V, attn.b_V)
        ]):
            rp = torch.einsum("bpm,nmh->bpnh", rel_ln, W)
            ip = torch.einsum("bpm,nmh->bpnh", irrel_ln, W)
            d = rp.abs() + ip.abs() + tol
            rp = rp + b * (rp.abs() / d)
            for h in range(n_heads):
                bwd = layer * (3 * n_heads + 1) + qkv_off * n_heads + h
                scores[src_fwd, bwd] += _l1(rp[:, :, h, :], pos_mask)

    def _propagate(rel, irrel, start_layer, src_fwd):
        r, ir = rel, irrel
        for l in range(start_layer, n_layers):
            blk = model.blocks[l]
            r_ln, ir_ln = cd_layer_norm(r, ir, blk.ln1.w, blk.ln1.b, eps=_get_eps(blk.ln1))
            _score_qkv(r_ln, ir_ln, l, src_fwd)
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal)
            r_mid = r + r_attn; ir_mid = ir + ir_attn; _normalize(r_mid, ir_mid)
            r_ln2, ir_ln2 = cd_layer_norm(r_mid, ir_mid, blk.ln2.w, blk.ln2.b, eps=_get_eps(blk.ln2))
            scores[src_fwd, l * (3 * n_heads + 1) + 3 * n_heads] += _l1(r_ln2, pos_mask)
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
            r = r_mid + r_mlp; ir = ir_mid + ir_mlp; _normalize(r, ir)

        ln_f = model.ln_final
        r_ln, ir_ln = cd_layer_norm(r, ir, ln_f.w, ln_f.b, eps=_get_eps(ln_f))
        r_logits, _ = cd_linear(r_ln, ir_ln, model.unembed.W_U, model.unembed.b_U)
        scores[src_fwd, -1] += _l1(r_logits, pos_mask)

    with torch.inference_mode():
        _propagate(embeddings, torch.zeros_like(embeddings), 0, 0)

        for l in range(n_layers):
            for h in range(n_heads):
                src_fwd = 1 + l * (n_heads + 1) + h
                head_out = head_outputs[l][:, :, h, :]
                rel = head_out
                irrel = resid_mid[l] - head_out

                blk = model.blocks[l]
                r_ln2, ir_ln2 = cd_layer_norm(rel, irrel, blk.ln2.w, blk.ln2.b, eps=_get_eps(blk.ln2))
                scores[src_fwd, l * (3 * n_heads + 1) + 3 * n_heads] += _l1(r_ln2, pos_mask)
                r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
                r_out = rel + r_mlp; ir_out = irrel + ir_mlp; _normalize(r_out, ir_out)
                _propagate(r_out, ir_out, l + 1, src_fwd)

            src_fwd = 1 + l * (n_heads + 1) + n_heads
            rel = mlp_outputs[l]
            irrel = resid_mid[l]
            _normalize(rel, irrel)
            _propagate(rel, irrel, l + 1, src_fwd)

    return scores


# ---------------------------------------------------------------------------
# Full CD-T decomposition (full=True)
# ---------------------------------------------------------------------------

def cd_edge_scores_full(
    model: HookedTransformer,
    tokens: Tensor,
    attention_mask: Tensor,
    input_lengths: Tensor,
    n_forward: int,
    n_backward: int,
    n_layers: int,
    n_heads: int,
) -> Tensor:
    """Edge scores using the full CD-T decomposition (Hsu et al., Algorithm 1).

    For each source node the decomposition starts from (rel=0, irrel=embeddings)
    and propagates through all layers.  At the source's layer, the source is
    marked as relevant in the attention value space (after softmax·V, before
    W_O), exactly as in the original paper.

    For MLP sources, the MLP output is marked after the MLP computation:
    rel = mlp_output, irrel is unchanged.

    Pre-layer activations are cached so layers before the source are skipped.
    """
    device = tokens.device
    dtype = model.cfg.dtype or torch.float32
    batch, n_pos = tokens.shape

    causal = _build_causal_mask(n_pos, attention_mask, device, dtype)
    pos_mask = (torch.arange(n_pos, device=device).unsqueeze(0)
                < input_lengths.unsqueeze(1)).to(dtype)

    scores = torch.zeros(n_forward, n_backward, device=device, dtype=torch.float32)

    # Phase 1: cache pre-layer residuals via a plain CD forward with rel=0
    with torch.inference_mode():
        embeddings = model.embed(tokens) + model.pos_embed(tokens)
        cached = [embeddings.clone()]   # cached[l] = residual at entry of layer l
        h_total = embeddings
        for l in range(n_layers):
            blk = model.blocks[l]
            r0 = torch.zeros_like(h_total)
            r_ln, ir_ln = cd_layer_norm(r0, h_total, blk.ln1.w, blk.ln1.b, eps=_get_eps(blk.ln1))
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal)
            h_mid = h_total + r_attn + ir_attn
            r_ln2, ir_ln2 = cd_layer_norm(r0[:1].expand_as(h_mid), h_mid,
                                          blk.ln2.w, blk.ln2.b, eps=_get_eps(blk.ln2))
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
            h_total = h_mid + r_mlp + ir_mlp
            cached.append(h_total.clone())

    # Scoring helpers (same as hook-based version)
    def _score_qkv(rel_ln, irrel_ln, layer, src_fwd):
        attn = model.blocks[layer].attn
        tol = 1e-8
        for qkv_off, (W, b) in enumerate([
            (attn.W_Q, attn.b_Q), (attn.W_K, attn.b_K), (attn.W_V, attn.b_V)
        ]):
            rp = torch.einsum("bpm,nmh->bpnh", rel_ln, W)
            ip = torch.einsum("bpm,nmh->bpnh", irrel_ln, W)
            d = rp.abs() + ip.abs() + tol
            rp = rp + b * (rp.abs() / d)
            for h_idx in range(n_heads):
                bwd = layer * (3 * n_heads + 1) + qkv_off * n_heads + h_idx
                scores[src_fwd, bwd] += _l1(rp[:, :, h_idx, :], pos_mask)

    def _propagate(rel, irrel, start_layer, src_fwd):
        """Propagate from start_layer onward, scoring destinations."""
        r, ir = rel, irrel
        for l in range(start_layer, n_layers):
            blk = model.blocks[l]
            r_ln, ir_ln = cd_layer_norm(r, ir, blk.ln1.w, blk.ln1.b, eps=_get_eps(blk.ln1))
            _score_qkv(r_ln, ir_ln, l, src_fwd)
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal)
            r_mid = r + r_attn; ir_mid = ir + ir_attn; _normalize(r_mid, ir_mid)
            r_ln2, ir_ln2 = cd_layer_norm(r_mid, ir_mid, blk.ln2.w, blk.ln2.b, eps=_get_eps(blk.ln2))
            scores[src_fwd, l * (3 * n_heads + 1) + 3 * n_heads] += _l1(r_ln2, pos_mask)
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
            r = r_mid + r_mlp; ir = ir_mid + ir_mlp; _normalize(r, ir)
        ln_f = model.ln_final
        r_ln, ir_ln = cd_layer_norm(r, ir, ln_f.w, ln_f.b, eps=_get_eps(ln_f))
        r_logits, _ = cd_linear(r_ln, ir_ln, model.unembed.W_U, model.unembed.b_U)
        scores[src_fwd, -1] += _l1(r_logits, pos_mask)

    with torch.inference_mode():
        # Source 0: input embeddings
        _propagate(embeddings, torch.zeros_like(embeddings), 0, 0)

        for l in range(n_layers):
            blk = model.blocks[l]

            # Start from cached residual at layer l: rel=0, irrel=cached[l]
            base_irrel = cached[l]

            # --- Attention head sources: full CD through layer l with marking ---
            for h in range(n_heads):
                src_fwd = 1 + l * (n_heads + 1) + h
                r = torch.zeros_like(base_irrel)
                ir = base_irrel.clone()

                # LN1 + Attention with head marking
                r_ln, ir_ln = cd_layer_norm(r, ir, blk.ln1.w, blk.ln1.b, eps=_get_eps(blk.ln1))
                r_attn, ir_attn = cd_attention_with_mark(
                    r_ln, ir_ln, blk, causal, mark_head=h)

                # Residual
                r_mid = r + r_attn
                ir_mid = ir + ir_attn
                _normalize(r_mid, ir_mid)

                # Score MLP destination at this layer
                r_ln2, ir_ln2 = cd_layer_norm(
                    r_mid, ir_mid, blk.ln2.w, blk.ln2.b, eps=_get_eps(blk.ln2))
                scores[src_fwd, l * (3 * n_heads + 1) + 3 * n_heads] += _l1(r_ln2, pos_mask)

                # MLP + residual
                r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
                r_out = r_mid + r_mlp
                ir_out = ir_mid + ir_mlp
                _normalize(r_out, ir_out)

                # Continue through remaining layers
                _propagate(r_out, ir_out, l + 1, src_fwd)

            # --- MLP source: full CD through layer l, mark MLP output ---
            src_fwd = 1 + l * (n_heads + 1) + n_heads
            r = torch.zeros_like(base_irrel)
            ir = base_irrel.clone()

            # LN1 + Attention (no marking — everything stays irrel through attn)
            r_ln, ir_ln = cd_layer_norm(r, ir, blk.ln1.w, blk.ln1.b, eps=_get_eps(blk.ln1))
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal)
            r_mid = r + r_attn
            ir_mid = ir + ir_attn
            _normalize(r_mid, ir_mid)

            # LN2 + MLP
            r_ln2, ir_ln2 = cd_layer_norm(
                r_mid, ir_mid, blk.ln2.w, blk.ln2.b, eps=_get_eps(blk.ln2))
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)

            # Mark: move total MLP output into rel
            mlp_total = r_mlp + ir_mlp
            r_out = r_mid + mlp_total
            ir_out = ir_mid                  # only pre-MLP residual in irrel
            _normalize(r_out, ir_out)

            _propagate(r_out, ir_out, l + 1, src_fwd)

    return scores

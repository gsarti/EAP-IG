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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
                  w, b, eps: float = 1e-5,
                  tol: float = 1e-8,
                  inv_std: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """LayerNorm decomposition (pre-norm GPT style).

    ``w`` and ``b`` may be ``None`` (e.g. ``LayerNormPre`` with ``fold_ln=True``),
    in which case the affine transform is skipped (equivalent to w=1, b=0).

    If ``inv_std`` is provided, it is reused instead of being recomputed from
    ``rel + irrel``.
    """
    r_mean = rel.mean(dim=-1, keepdim=True)
    ir_mean = irrel.mean(dim=-1, keepdim=True)

    if inv_std is None:
        tot = rel + irrel
        var = tot.pow(2).mean(-1, keepdim=True) - tot.mean(-1, keepdim=True).pow(2)
        inv_std = (var + eps).rsqrt()

    r_out = (rel - r_mean) * inv_std
    ir_out = (irrel - ir_mean) * inv_std

    if w is not None:
        r_out = r_out * w
        ir_out = ir_out * w

    if b is not None:
        frac = r_out.abs() / (r_out.abs() + ir_out.abs() + tol)
        r_out = r_out + b * frac
        ir_out = ir_out + b * (1 - frac)

    return r_out, ir_out


def cd_gelu(rel: Tensor, irrel: Tensor) -> Tuple[Tensor, Tensor]:
    """GELU decomposition (ACD rule)."""
    ir_act = nn.functional.gelu(irrel)
    return nn.functional.gelu(rel + irrel) - ir_act, ir_act


def _get_eps(ln_module) -> float:
    if hasattr(ln_module, 'cfg') and hasattr(ln_module.cfg, 'eps'):
        return ln_module.cfg.eps
    return 1e-5


def _ln_params(ln_module):
    """Return (w, b, eps) from a LayerNorm or LayerNormPre module."""
    w = getattr(ln_module, 'w', None)
    b = getattr(ln_module, 'b', None)
    return w, b, _get_eps(ln_module)


# ---------------------------------------------------------------------------
# Attention decomposition
# ---------------------------------------------------------------------------

def cd_attention(
    rel: Tensor, irrel: Tensor, block, causal_mask: Tensor,
    cache: Optional['LayerCache'] = None,
) -> Tuple[Tensor, Tensor]:
    """Decompose self-attention (no LN, no residual).

    If *cache* is provided, reuses precomputed ``tot_probs`` and ``tv``/
    ``attn_ctx`` from the clean forward pass.  Cache tensors are broadcast
    along the batch dimension when the batch sizes differ (batched sources).
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

    rel_scores = torch.einsum("bqnh,bknh->bnqk", rq, rk) / scale + causal_mask
    rel_probs = nn.functional.softmax(rel_scores, dim=-1)

    if cache is not None:
        tot_ctx = cache.attn_ctx
        # Broadcast cache if needed (batched sources)
        if tot_ctx.shape[0] != rel.shape[0]:
            n_src = rel.shape[0] // tot_ctx.shape[0]
            tot_ctx = tot_ctx.repeat(n_src, 1, 1, 1)
    else:
        tq, tk = rq + iq, rk + ik
        tot_scores = torch.einsum("bqnh,bknh->bnqk", tq, tk) / scale + causal_mask
        tot_probs = nn.functional.softmax(tot_scores, dim=-1)
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
    cache: Optional['LayerCache'] = None,
) -> Tuple[Tensor, Tensor]:
    """Decompose self-attention with optional source marking in value space.

    If ``mark_head >= 0``, after computing softmax·V (in d_head space per
    head), the specified head's irrel is moved to rel — matching the original
    CD-T paper's ``set_rel_at_source_nodes`` which operates between the
    attention mechanism and the W_O projection.
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

    rel_scores = torch.einsum("bqnh,bknh->bnqk", rq, rk) / scale + causal_mask
    rel_probs = nn.functional.softmax(rel_scores, dim=-1)

    if cache is not None:
        tot_ctx = cache.attn_ctx
        if tot_ctx.shape[0] != rel.shape[0]:
            n_src = rel.shape[0] // tot_ctx.shape[0]
            tot_ctx = tot_ctx.repeat(n_src, 1, 1, 1)
    else:
        tq, tk = rq + iq, rk + ik
        tot_scores = torch.einsum("bqnh,bknh->bnqk", tq, tk) / scale + causal_mask
        tot_probs = nn.functional.softmax(tot_scores, dim=-1)
        tv = rv + iv
        tot_ctx = torch.einsum("bnqk,bknh->bqnh", tot_probs, tv)

    rel_ctx = torch.einsum("bnqk,bknh->bqnh", rel_probs, rv)
    irrel_ctx = tot_ctx - rel_ctx

    if mark_head >= 0:
        h = mark_head
        rel_ctx[:, :, h, :] = rel_ctx[:, :, h, :] + irrel_ctx[:, :, h, :]
        irrel_ctx[:, :, h, :] = 0

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
# Clean forward pass cache
# ---------------------------------------------------------------------------

@dataclass
class LayerCache:
    """Cached clean forward pass results for one transformer layer."""
    ln1_inv_std: Tensor        # (batch, pos, 1)
    attn_probs: Tensor         # (batch, n_heads, pos, pos)
    tv: Tensor                 # (batch, pos, n_heads, d_head)
    attn_ctx: Tensor           # (batch, pos, n_heads, d_head)
    attn_out: Tensor           # (batch, pos, d_model)
    ln2_inv_std: Tensor        # (batch, pos, 1)
    mlp_out: Tensor            # (batch, pos, d_model)


@dataclass
class ForwardCache:
    """All cached results from the clean forward pass."""
    embeddings: Tensor
    layers: List[LayerCache] = field(default_factory=list)
    resid_pre: List[Tensor] = field(default_factory=list)
    resid_mid: List[Tensor] = field(default_factory=list)
    head_outputs: List[Tensor] = field(default_factory=list)  # (batch, pos, n_heads, d_model)
    ln_final_inv_std: Optional[Tensor] = None


def _compute_ln_inv_std(x: Tensor, eps: float) -> Tensor:
    var = x.pow(2).mean(-1, keepdim=True) - x.mean(-1, keepdim=True).pow(2)
    return (var + eps).rsqrt()


def _cache_clean_forward(
    model: HookedTransformer, tokens: Tensor, attention_mask: Tensor,
    causal_mask: Tensor, n_layers: int,
) -> ForwardCache:
    """Run one clean forward pass and cache everything needed for CD reuse."""
    embeddings = model.embed(tokens) + model.pos_embed(tokens)
    fc = ForwardCache(embeddings=embeddings)

    h = embeddings
    for l in range(n_layers):
        blk = model.blocks[l]
        fc.resid_pre.append(h)

        w1, b1, eps1 = _ln_params(blk.ln1)
        ln1_inv_std = _compute_ln_inv_std(h, eps1)
        h_mean = h.mean(-1, keepdim=True)
        h_ln = (h - h_mean) * ln1_inv_std
        if w1 is not None:
            h_ln = h_ln * w1
        if b1 is not None:
            h_ln = h_ln + b1

        attn = blk.attn
        scale = math.sqrt(attn.cfg.d_head)
        tq = torch.einsum("bpm,nmh->bpnh", h_ln, attn.W_Q) + attn.b_Q
        tk = torch.einsum("bpm,nmh->bpnh", h_ln, attn.W_K) + attn.b_K
        tv = torch.einsum("bpm,nmh->bpnh", h_ln, attn.W_V) + attn.b_V
        tot_scores = torch.einsum("bqnh,bknh->bnqk", tq, tk) / scale + causal_mask
        tot_probs = nn.functional.softmax(tot_scores, dim=-1)
        attn_ctx = torch.einsum("bnqk,bknh->bqnh", tot_probs, tv)
        attn_out = torch.einsum("bqnh,nhm->bqm", attn_ctx, attn.W_O) + attn.b_O

        head_out_per_head = torch.einsum("bqnh,nhm->bqnm", attn_ctx, attn.W_O)
        fc.head_outputs.append(head_out_per_head)

        h_mid = h + attn_out
        fc.resid_mid.append(h_mid)

        w2, b2, eps2 = _ln_params(blk.ln2)
        ln2_inv_std = _compute_ln_inv_std(h_mid, eps2)
        h_mid_mean = h_mid.mean(-1, keepdim=True)
        h_ln2 = (h_mid - h_mid_mean) * ln2_inv_std
        if w2 is not None:
            h_ln2 = h_ln2 * w2
        if b2 is not None:
            h_ln2 = h_ln2 + b2

        mlp = blk.mlp
        h_mlp = h_ln2 @ mlp.W_in + mlp.b_in
        h_mlp = nn.functional.gelu(h_mlp)
        mlp_out = h_mlp @ mlp.W_out + mlp.b_out

        fc.layers.append(LayerCache(
            ln1_inv_std=ln1_inv_std,
            attn_probs=tot_probs,
            tv=tv,
            attn_ctx=attn_ctx,
            attn_out=attn_out,
            ln2_inv_std=ln2_inv_std,
            mlp_out=mlp_out,
        ))

        h = h_mid + mlp_out

    ln_f = model.ln_final
    fc.ln_final_inv_std = _compute_ln_inv_std(h, _get_eps(ln_f))

    return fc


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


def _expand_cache_inv_std(inv_std: Tensor, n_src: int) -> Tensor:
    """Repeat cached inv_std for n_src batched sources."""
    return inv_std.repeat(n_src, 1, 1)


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
    """Edge scores using hook-based source extraction + CD propagation.

    Sources from the same layer are batched together: their (rel, irrel)
    tensors are stacked along the batch dimension and propagated in one pass,
    giving ~Nx throughput from a single set of matmuls instead of N serial ones.
    """
    device = tokens.device
    dtype = model.cfg.dtype or torch.float32
    batch, n_pos = tokens.shape

    causal = _build_causal_mask(n_pos, attention_mask, device, dtype)
    pos_mask = (torch.arange(n_pos, device=device).unsqueeze(0)
                < input_lengths.unsqueeze(1)).to(dtype)

    scores = torch.zeros(n_forward, n_backward, device=device, dtype=torch.float32)

    with torch.inference_mode():
        fc = _cache_clean_forward(model, tokens, attention_mask, causal, n_layers)
        embeddings = fc.embeddings

    # Batched scoring: rel_ln has shape (n_src * batch, pos, d_model).
    # We reshape to (n_src, batch, pos, ...) to extract per-source scores.
    def _score_qkv_batched(rel_ln, irrel_ln, layer, src_fwd_list, n_src, pm):
        attn = model.blocks[layer].attn
        tol = 1e-8
        for qkv_off, (W, b) in enumerate([
            (attn.W_Q, attn.b_Q), (attn.W_K, attn.b_K), (attn.W_V, attn.b_V)
        ]):
            rp = torch.einsum("bpm,nmh->bpnh", rel_ln, W)
            ip = torch.einsum("bpm,nmh->bpnh", irrel_ln, W)
            d = rp.abs() + ip.abs() + tol
            rp = rp + b * (rp.abs() / d)
            # (n_src * batch, pos, n_heads, d_head) → (n_src, batch, pos, n_heads, d_head)
            rp_5d = rp.view(n_src, batch, n_pos, n_heads, -1)
            # L1 per source per head: sum d_head, mask, sum batch+pos
            l1 = (rp_5d.abs().sum(dim=-1) * pm.unsqueeze(-1)).sum(dim=(1, 2))  # (n_src, n_heads)
            for h in range(n_heads):
                bwd = layer * (3 * n_heads + 1) + qkv_off * n_heads + h
                for si, src_fwd in enumerate(src_fwd_list):
                    scores[src_fwd, bwd] += l1[si, h].item()

    def _score_qkv_single(rel_ln, irrel_ln, layer, src_fwd):
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

    def _propagate_batched(rel, irrel, start_layer, src_fwd_list, n_src, pm):
        """Propagate n_src sources batched along dim 0.
        rel/irrel: (n_src * batch, pos, d_model)
        pm: (n_src * batch, pos) — batched pos_mask
        """
        r, ir = rel, irrel
        for l in range(start_layer, n_layers):
            blk = model.blocks[l]
            lc = fc.layers[l]
            w1, b1, eps1 = _ln_params(blk.ln1)
            inv1 = _expand_cache_inv_std(lc.ln1_inv_std, n_src)
            r_ln, ir_ln = cd_layer_norm(r, ir, w1, b1, eps1, inv_std=inv1)
            _score_qkv_batched(r_ln, ir_ln, l, src_fwd_list, n_src, pm)
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal, cache=lc)
            r_mid = r + r_attn; ir_mid = ir + ir_attn; _normalize(r_mid, ir_mid)
            w2, b2, eps2 = _ln_params(blk.ln2)
            inv2 = _expand_cache_inv_std(lc.ln2_inv_std, n_src)
            r_ln2, ir_ln2 = cd_layer_norm(r_mid, ir_mid, w2, b2, eps2, inv_std=inv2)
            # MLP destination score per source
            r_ln2_5d = r_ln2.view(n_src, batch, n_pos, -1)
            l1_mlp = (r_ln2_5d.abs().sum(dim=-1) * pm.view(n_src, batch, n_pos)).sum(dim=(1, 2))
            mlp_bwd = l * (3 * n_heads + 1) + 3 * n_heads
            for si, src_fwd in enumerate(src_fwd_list):
                scores[src_fwd, mlp_bwd] += l1_mlp[si].item()
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
            r = r_mid + r_mlp; ir = ir_mid + ir_mlp; _normalize(r, ir)

        ln_f = model.ln_final
        w_f, b_f, eps_f = _ln_params(ln_f)
        inv_f = _expand_cache_inv_std(fc.ln_final_inv_std, n_src)
        r_ln, ir_ln = cd_layer_norm(r, ir, w_f, b_f, eps_f, inv_std=inv_f)
        r_logits, _ = cd_linear(r_ln, ir_ln, model.unembed.W_U, model.unembed.b_U)
        r_logits_5d = r_logits.view(n_src, batch, n_pos, -1)
        l1_logits = (r_logits_5d.abs().sum(dim=-1) * pm.view(n_src, batch, n_pos)).sum(dim=(1, 2))
        for si, src_fwd in enumerate(src_fwd_list):
            scores[src_fwd, -1] += l1_logits[si].item()

    def _propagate_single(rel, irrel, start_layer, src_fwd):
        r, ir = rel, irrel
        for l in range(start_layer, n_layers):
            blk = model.blocks[l]
            lc = fc.layers[l]
            w1, b1, eps1 = _ln_params(blk.ln1)
            r_ln, ir_ln = cd_layer_norm(r, ir, w1, b1, eps1, inv_std=lc.ln1_inv_std)
            _score_qkv_single(r_ln, ir_ln, l, src_fwd)
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal, cache=lc)
            r_mid = r + r_attn; ir_mid = ir + ir_attn; _normalize(r_mid, ir_mid)
            w2, b2, eps2 = _ln_params(blk.ln2)
            r_ln2, ir_ln2 = cd_layer_norm(r_mid, ir_mid, w2, b2, eps2, inv_std=lc.ln2_inv_std)
            scores[src_fwd, l * (3 * n_heads + 1) + 3 * n_heads] += _l1(r_ln2, pos_mask)
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
            r = r_mid + r_mlp; ir = ir_mid + ir_mlp; _normalize(r, ir)

        ln_f = model.ln_final
        w_f, b_f, eps_f = _ln_params(ln_f)
        r_ln, ir_ln = cd_layer_norm(r, ir, w_f, b_f, eps_f, inv_std=fc.ln_final_inv_std)
        r_logits, _ = cd_linear(r_ln, ir_ln, model.unembed.W_U, model.unembed.b_U)
        scores[src_fwd, -1] += _l1(r_logits, pos_mask)

    with torch.inference_mode():
        # Source 0: input embeddings (single propagation through all layers)
        _propagate_single(embeddings, torch.zeros_like(embeddings), 0, 0)

        for l in range(n_layers):
            blk = model.blocks[l]
            lc = fc.layers[l]

            # Prepare all head sources at this layer
            rels_list = []
            irrels_list = []
            src_fwd_list = []
            for h in range(n_heads):
                src_fwd = 1 + l * (n_heads + 1) + h
                head_out = fc.head_outputs[l][:, :, h, :]
                rel_h = head_out
                irrel_h = fc.resid_mid[l] - head_out

                # Score MLP destination at this layer for each head source
                w2, b2, eps2 = _ln_params(blk.ln2)
                r_ln2, ir_ln2 = cd_layer_norm(rel_h, irrel_h, w2, b2, eps2, inv_std=lc.ln2_inv_std)
                scores[src_fwd, l * (3 * n_heads + 1) + 3 * n_heads] += _l1(r_ln2, pos_mask)

                # Prepare for batched propagation
                r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
                r_out = rel_h + r_mlp; ir_out = irrel_h + ir_mlp; _normalize(r_out, ir_out)
                rels_list.append(r_out)
                irrels_list.append(ir_out)
                src_fwd_list.append(src_fwd)

            # Add MLP source
            mlp_src_fwd = 1 + l * (n_heads + 1) + n_heads
            rel_mlp = fc.layers[l].mlp_out.clone()
            irrel_mlp = fc.resid_mid[l].clone()
            _normalize(rel_mlp, irrel_mlp)
            rels_list.append(rel_mlp)
            irrels_list.append(irrel_mlp)
            src_fwd_list.append(mlp_src_fwd)

            # Batch all sources from this layer
            n_src = len(src_fwd_list)
            if l + 1 < n_layers:  # only propagate if there are more layers
                batched_rel = torch.cat(rels_list, dim=0)   # (n_src * batch, pos, d_model)
                batched_irrel = torch.cat(irrels_list, dim=0)
                pm = pos_mask.repeat(n_src, 1)  # (n_src * batch, pos)
                _propagate_batched(batched_rel, batched_irrel, l + 1, src_fwd_list, n_src, pm)
            else:
                # Last layer: only logits destination
                ln_f = model.ln_final
                w_f, b_f, eps_f = _ln_params(ln_f)
                for si, src_fwd in enumerate(src_fwd_list):
                    r_ln, ir_ln = cd_layer_norm(
                        rels_list[si], irrels_list[si], w_f, b_f, eps_f,
                        inv_std=fc.ln_final_inv_std)
                    r_logits, _ = cd_linear(r_ln, ir_ln, model.unembed.W_U, model.unembed.b_U)
                    scores[src_fwd, -1] += _l1(r_logits, pos_mask)

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

    Uses batched propagation: all heads + MLP at the same layer are processed
    together through subsequent layers.
    """
    device = tokens.device
    dtype = model.cfg.dtype or torch.float32
    batch, n_pos = tokens.shape

    causal = _build_causal_mask(n_pos, attention_mask, device, dtype)
    pos_mask = (torch.arange(n_pos, device=device).unsqueeze(0)
                < input_lengths.unsqueeze(1)).to(dtype)

    scores = torch.zeros(n_forward, n_backward, device=device, dtype=torch.float32)

    with torch.inference_mode():
        fc = _cache_clean_forward(model, tokens, attention_mask, causal, n_layers)
        embeddings = fc.embeddings

    def _score_qkv_batched(rel_ln, irrel_ln, layer, src_fwd_list, n_src, pm):
        attn = model.blocks[layer].attn
        tol = 1e-8
        for qkv_off, (W, b) in enumerate([
            (attn.W_Q, attn.b_Q), (attn.W_K, attn.b_K), (attn.W_V, attn.b_V)
        ]):
            rp = torch.einsum("bpm,nmh->bpnh", rel_ln, W)
            ip = torch.einsum("bpm,nmh->bpnh", irrel_ln, W)
            d = rp.abs() + ip.abs() + tol
            rp = rp + b * (rp.abs() / d)
            rp_5d = rp.view(n_src, batch, n_pos, n_heads, -1)
            l1 = (rp_5d.abs().sum(dim=-1) * pm.unsqueeze(-1)).sum(dim=(1, 2))
            for h in range(n_heads):
                bwd = layer * (3 * n_heads + 1) + qkv_off * n_heads + h
                for si, src_fwd in enumerate(src_fwd_list):
                    scores[src_fwd, bwd] += l1[si, h].item()

    def _score_qkv_single(rel_ln, irrel_ln, layer, src_fwd):
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

    def _propagate_batched(rel, irrel, start_layer, src_fwd_list, n_src, pm):
        r, ir = rel, irrel
        for l in range(start_layer, n_layers):
            blk = model.blocks[l]
            lc = fc.layers[l]
            w1, b1, eps1 = _ln_params(blk.ln1)
            inv1 = _expand_cache_inv_std(lc.ln1_inv_std, n_src)
            r_ln, ir_ln = cd_layer_norm(r, ir, w1, b1, eps1, inv_std=inv1)
            _score_qkv_batched(r_ln, ir_ln, l, src_fwd_list, n_src, pm)
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal, cache=lc)
            r_mid = r + r_attn; ir_mid = ir + ir_attn; _normalize(r_mid, ir_mid)
            w2, b2, eps2 = _ln_params(blk.ln2)
            inv2 = _expand_cache_inv_std(lc.ln2_inv_std, n_src)
            r_ln2, ir_ln2 = cd_layer_norm(r_mid, ir_mid, w2, b2, eps2, inv_std=inv2)
            r_ln2_5d = r_ln2.view(n_src, batch, n_pos, -1)
            l1_mlp = (r_ln2_5d.abs().sum(dim=-1) * pm.view(n_src, batch, n_pos)).sum(dim=(1, 2))
            mlp_bwd = l * (3 * n_heads + 1) + 3 * n_heads
            for si, src_fwd in enumerate(src_fwd_list):
                scores[src_fwd, mlp_bwd] += l1_mlp[si].item()
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
            r = r_mid + r_mlp; ir = ir_mid + ir_mlp; _normalize(r, ir)

        ln_f = model.ln_final
        w_f, b_f, eps_f = _ln_params(ln_f)
        inv_f = _expand_cache_inv_std(fc.ln_final_inv_std, n_src)
        r_ln, ir_ln = cd_layer_norm(r, ir, w_f, b_f, eps_f, inv_std=inv_f)
        r_logits, _ = cd_linear(r_ln, ir_ln, model.unembed.W_U, model.unembed.b_U)
        r_logits_5d = r_logits.view(n_src, batch, n_pos, -1)
        l1_logits = (r_logits_5d.abs().sum(dim=-1) * pm.view(n_src, batch, n_pos)).sum(dim=(1, 2))
        for si, src_fwd in enumerate(src_fwd_list):
            scores[src_fwd, -1] += l1_logits[si].item()

    def _propagate_single(rel, irrel, start_layer, src_fwd):
        r, ir = rel, irrel
        for l in range(start_layer, n_layers):
            blk = model.blocks[l]
            lc = fc.layers[l]
            w1, b1, eps1 = _ln_params(blk.ln1)
            r_ln, ir_ln = cd_layer_norm(r, ir, w1, b1, eps1, inv_std=lc.ln1_inv_std)
            _score_qkv_single(r_ln, ir_ln, l, src_fwd)
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal, cache=lc)
            r_mid = r + r_attn; ir_mid = ir + ir_attn; _normalize(r_mid, ir_mid)
            w2, b2, eps2 = _ln_params(blk.ln2)
            r_ln2, ir_ln2 = cd_layer_norm(r_mid, ir_mid, w2, b2, eps2, inv_std=lc.ln2_inv_std)
            scores[src_fwd, l * (3 * n_heads + 1) + 3 * n_heads] += _l1(r_ln2, pos_mask)
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
            r = r_mid + r_mlp; ir = ir_mid + ir_mlp; _normalize(r, ir)
        ln_f = model.ln_final
        w_f, b_f, eps_f = _ln_params(ln_f)
        r_ln, ir_ln = cd_layer_norm(r, ir, w_f, b_f, eps_f, inv_std=fc.ln_final_inv_std)
        r_logits, _ = cd_linear(r_ln, ir_ln, model.unembed.W_U, model.unembed.b_U)
        scores[src_fwd, -1] += _l1(r_logits, pos_mask)

    with torch.inference_mode():
        _propagate_single(embeddings, torch.zeros_like(embeddings), 0, 0)

        for l in range(n_layers):
            blk = model.blocks[l]
            lc = fc.layers[l]
            base_irrel = fc.resid_pre[l]

            rels_list = []
            irrels_list = []
            src_fwd_list = []

            # Attention head sources
            for h in range(n_heads):
                src_fwd = 1 + l * (n_heads + 1) + h
                r = torch.zeros_like(base_irrel)
                ir = base_irrel.clone()

                w1, b1, eps1 = _ln_params(blk.ln1)
                r_ln, ir_ln = cd_layer_norm(r, ir, w1, b1, eps1, inv_std=lc.ln1_inv_std)
                r_attn, ir_attn = cd_attention_with_mark(
                    r_ln, ir_ln, blk, causal, mark_head=h, cache=lc)

                r_mid = r + r_attn
                ir_mid = ir + ir_attn
                _normalize(r_mid, ir_mid)

                w2, b2, eps2 = _ln_params(blk.ln2)
                r_ln2, ir_ln2 = cd_layer_norm(
                    r_mid, ir_mid, w2, b2, eps2, inv_std=lc.ln2_inv_std)
                scores[src_fwd, l * (3 * n_heads + 1) + 3 * n_heads] += _l1(r_ln2, pos_mask)

                r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)
                r_out = r_mid + r_mlp
                ir_out = ir_mid + ir_mlp
                _normalize(r_out, ir_out)

                rels_list.append(r_out)
                irrels_list.append(ir_out)
                src_fwd_list.append(src_fwd)

            # MLP source
            mlp_src_fwd = 1 + l * (n_heads + 1) + n_heads
            r = torch.zeros_like(base_irrel)
            ir = base_irrel.clone()

            w1, b1, eps1 = _ln_params(blk.ln1)
            r_ln, ir_ln = cd_layer_norm(r, ir, w1, b1, eps1, inv_std=lc.ln1_inv_std)
            r_attn, ir_attn = cd_attention(r_ln, ir_ln, blk, causal, cache=lc)
            r_mid = r + r_attn
            ir_mid = ir + ir_attn
            _normalize(r_mid, ir_mid)

            w2, b2, eps2 = _ln_params(blk.ln2)
            r_ln2, ir_ln2 = cd_layer_norm(
                r_mid, ir_mid, w2, b2, eps2, inv_std=lc.ln2_inv_std)
            r_mlp, ir_mlp = cd_mlp(r_ln2, ir_ln2, blk)

            mlp_total = r_mlp + ir_mlp
            r_out = r_mid + mlp_total
            ir_out = ir_mid
            _normalize(r_out, ir_out)

            rels_list.append(r_out)
            irrels_list.append(ir_out)
            src_fwd_list.append(mlp_src_fwd)

            # Batch propagation through remaining layers
            n_src = len(src_fwd_list)
            if l + 1 < n_layers:
                batched_rel = torch.cat(rels_list, dim=0)
                batched_irrel = torch.cat(irrels_list, dim=0)
                pm = pos_mask.repeat(n_src, 1)
                _propagate_batched(batched_rel, batched_irrel, l + 1, src_fwd_list, n_src, pm)
            else:
                ln_f = model.ln_final
                w_f, b_f, eps_f = _ln_params(ln_f)
                for si, src_fwd in enumerate(src_fwd_list):
                    r_ln, ir_ln = cd_layer_norm(
                        rels_list[si], irrels_list[si], w_f, b_f, eps_f,
                        inv_std=fc.ln_final_inv_std)
                    r_logits, _ = cd_linear(r_ln, ir_ln, model.unembed.W_U, model.unembed.b_U)
                    scores[src_fwd, -1] += _l1(r_logits, pos_mask)

    return scores

"""PF-GIM: Proximity-Filtered GIM for circuit discovery.

Computes GIM-corrected gradient scores (activation_diff × GIM_grad) for all
edges, then filters out structurally implausible edges using a structural
heuristic. Edges with heuristic score below a quantile threshold are zeroed
out. Gradient provides ranking, the heuristic provides structural filtering.

Scoring functions for the filtering heuristics:
  - compute_proximity_scores: ALTI proximity (default)
  - compute_norm_scores: L1 norm of source contributions
  - compute_cosine_scores: cosine similarity to destination residual
  - compute_logit_scores: logit-space importance via unembedding projection
    (Ferrando et al., ACL 2023)
"""

from typing import Optional

import torch
from torch import Tensor


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


def compute_norm_scores(
    contributions: Tensor,
    input_lengths: Tensor,
) -> Tensor:
    """L1 norm of source contributions as importance proxy.

    contributions: (batch, pos, n_src, d_model)
    input_lengths: (batch,)

    Returns: (n_src,) scores aggregated over batch and positions.
    """
    norms = torch.linalg.vector_norm(contributions, ord=1, dim=-1)

    max_len = input_lengths.max()
    mask = torch.arange(max_len, device=input_lengths.device, dtype=input_lengths.dtype
                        ).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
    norms = norms * mask.unsqueeze(-1)

    norms = norms.sum(dim=1) / input_lengths.view(-1, 1)
    return norms.sum(dim=0)


def compute_cosine_scores(
    contributions: Tensor,
    reference: Tensor,
    input_lengths: Tensor,
) -> Tensor:
    """Cosine similarity between contributions and reference (clamped >= 0).

    contributions: (batch, pos, n_src, d_model)
    reference: (batch, pos, d_model)
    input_lengths: (batch,)

    Returns: (n_src,) scores aggregated over batch and positions.
    """
    cos = torch.nn.functional.cosine_similarity(
        contributions, reference.unsqueeze(2), dim=-1)
    cos = torch.clamp(cos, min=0)

    max_len = input_lengths.max()
    mask = torch.arange(max_len, device=input_lengths.device, dtype=input_lengths.dtype
                        ).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
    cos = cos * mask.unsqueeze(-1)

    cos = cos.sum(dim=1) / input_lengths.view(-1, 1)
    return cos.sum(dim=0)


def compute_logit_scores(
    contributions: Tensor,
    W_U: Tensor,
    input_lengths: Tensor,
    target_tokens: Tensor,
    foil_tokens: Optional[Tensor] = None,
) -> Tensor:
    """Logit-space importance: contrastive contribution to predicted token's logit.

    For each source, computes how much its contribution at the output position
    pushes the logit of the predicted token (and away from the foil token if
    provided), following Ferrando et al. (ACL 2023). Uses the dot product with
    the contrastive unembedding direction (target - foil).

    contributions: (batch, pos, n_src, d_model)
    W_U: (d_model, d_vocab)
    input_lengths: (batch,)
    target_tokens: (batch,) clean predicted token indices
    foil_tokens: (batch,) corrupted predicted token indices (optional)

    Returns: (n_src,) scores aggregated over batch.
    """
    batch_size = contributions.shape[0]
    device = contributions.device

    # Get output position contributions: (batch, n_src, d_model)
    output_idx = input_lengths - 1
    output_contribs = contributions[
        torch.arange(batch_size, device=device), output_idx
    ]

    # Contrastive unembedding direction: (batch, d_model)
    direction = W_U[:, target_tokens].T
    if foil_tokens is not None:
        direction = direction - W_U[:, foil_tokens].T

    # Dot product: how much each source pushes along the contrastive direction
    logit_contribs = torch.einsum('bsd,bd->bs', output_contribs, direction)

    return logit_contribs.abs().sum(dim=0)


def compute_local_mixing_weights(
    contributions: Tensor,
    reference: Tensor,
) -> Tensor:
    """Compute per-position, per-batch ALTI proximity mixing weights.

    Unlike compute_proximity_scores, does NOT aggregate over batch/position.
    Returns the raw normalized importance weights at each position.

    contributions: (batch, pos, n_src, d_model)
    reference:     (batch, pos, d_model)

    Returns: (batch, pos, n_src) mixing weights summing to 1 over n_src dim.
    """
    ref_unsq = reference.unsqueeze(2)
    dist = torch.linalg.vector_norm(contributions - ref_unsq, ord=1, dim=-1)
    ref_norm = torch.linalg.vector_norm(ref_unsq, ord=1, dim=-1)
    proximity = torch.clamp(-dist + ref_norm, min=0)
    prox_sum = proximity.sum(dim=2, keepdim=True).clamp(min=1e-10)
    return proximity / prox_sum


def compose_mixing_weights(
    local_mixing: list,
    n_forward: int,
    batch_size: int,
    n_pos: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Compose local ALTI mixing weights across the graph in topological order.

    For each source node s, composed[:, :, s] tracks how much of the
    information flowing through the network traces back to s, accounting
    for dilution through all subsequent layers.

    After composing all intermediate nodes, applies the terminal (logits)
    destination's mixing weights to produce final per-source importance:
    final[s] = logit_mixing[s] * composed[s].

    Args:
        local_mixing: list of (fwd_key, prev_index, weights) tuples.
            fwd_key is an int for MLP outputs, a (start, stop) tuple for
            attention outputs (broadcast to all heads), or -1 for the
            logits terminal destination.
            weights is (batch, pos, prev_index).
        n_forward: total number of forward (source) nodes
        batch_size, n_pos, device, dtype: tensor allocation parameters

    Returns:
        (batch, pos, n_forward) composed importance weights
    """
    from collections import defaultdict

    composed = torch.zeros((batch_size, n_pos, n_forward), device=device, dtype=dtype)
    composed[:, :, 0] = 1.0  # input node starts with weight 1.0

    # Group by fwd_key and average Q/K/V entries for attention nodes
    grouped = defaultdict(list)
    for fwd_key, prev_idx, weights in local_mixing:
        grouped[fwd_key].append((prev_idx, weights))

    # Separate logits sentinel from normal nodes
    logits_entries = grouped.pop(-1, None)

    # Process source nodes in topological order
    sort_key = lambda k: k[0] if isinstance(k, tuple) else k
    for fwd_key in sorted(grouped.keys(), key=sort_key):
        entries = grouped[fwd_key]
        prev_idx = entries[0][0]
        avg_weights = torch.stack([w for _, w in entries]).mean(dim=0)
        # avg_weights: (batch, pos, prev_idx)
        input_composed = composed[:, :, :prev_idx]
        value = (avg_weights * input_composed).sum(dim=2)  # (batch, pos)

        if isinstance(fwd_key, tuple):
            # Attention: broadcast to all head indices
            start, stop = fwd_key
            composed[:, :, start:stop] = value.unsqueeze(-1).expand(
                -1, -1, stop - start)
        else:
            composed[:, :, fwd_key] = value

    # Apply logits destination mixing for final per-source importance
    if logits_entries is not None:
        prev_idx = logits_entries[0][0]
        avg_weights = torch.stack([w for _, w in logits_entries]).mean(dim=0)
        composed[:, :, :prev_idx] = avg_weights * composed[:, :, :prev_idx]
        if prev_idx < n_forward:
            composed[:, :, prev_idx:] = 0.0

    return composed


def compute_propagated_logit_scores(
    source_acts: Tensor,
    composed_weights: Tensor,
    W_U: Tensor,
    input_lengths: Tensor,
    target_tokens: Tensor,
    foil_tokens: Optional[Tensor] = None,
) -> Tensor:
    """Logit-space importance using ALTI-propagated contributions.

    For each source s, computes the composed importance c_s (how much of the
    final representation traces back to s), then weights the logit projection
    of s's activation by c_s. This accounts for dilution/amplification through
    subsequent layers.

    source_acts:      (batch, pos, n_src, d_model) raw source activations
    composed_weights: (batch, pos, n_src) composed importance weights
    W_U:              (d_model, d_vocab)
    input_lengths:    (batch,)
    target_tokens:    (batch,)
    foil_tokens:      (batch,) optional

    Returns: (n_src,) scores aggregated over batch.
    """
    batch_size = source_acts.shape[0]
    device = source_acts.device

    # Get output position data
    output_idx = input_lengths - 1
    batch_arange = torch.arange(batch_size, device=device)
    output_contribs = source_acts[batch_arange, output_idx]      # (batch, n_src, d_model)
    output_weights = composed_weights[batch_arange, output_idx]  # (batch, n_src)

    # Contrastive unembedding direction: (batch, d_model)
    direction = W_U[:, target_tokens].T
    if foil_tokens is not None:
        direction = direction - W_U[:, foil_tokens].T

    # Raw logit contribution per source, weighted by composed importance
    raw_logit = torch.einsum('bsd,bd->bs', output_contribs, direction)
    propagated = output_weights * raw_logit.abs()

    return propagated.sum(dim=0)

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

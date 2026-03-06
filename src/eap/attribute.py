from typing import Callable, List, Optional, Literal, Tuple
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer

from tqdm import tqdm

from .utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations
from .evaluate import evaluate_graph, evaluate_baseline
from .graph import Graph
from .gwai import (
    compute_edge_gradient_scores, compute_combined_scores,
    compute_proximity_scores, make_names_filter,
    _propagate_chunked_attention, _propagate_chunked_mlp,
)


def _model_device(model: HookedTransformer):
    return next(model.parameters()).device

def get_scores_exact(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
                     intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                     intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    """Gets scores via exact patching, by repeatedly calling evaluate graph.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): the intervention to use. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): the dataloader over which to take the mean. Defaults to None.
        quiet (bool, optional): _description_. Defaults to False.
    """

    graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
    baseline = evaluate_baseline(model, dataloader, metric).mean().item()
    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
        edge.in_graph = False
        intervened_performance = evaluate_graph(model, graph, dataloader, metric, intervention=intervention, intervention_dataloader=intervention_dataloader, 
                                                quiet=True, skip_clean=True).mean().item()
        edge.score = intervened_performance - baseline
        edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores


def get_scores_eap(model: HookedTransformer, graph: Graph, dataloader:DataLoader, metric: Callable[[Tensor], Tensor], 
                   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                   intervention_dataloader: Optional[DataLoader]=None, quiet=False):
    """Gets edge attribution scores using EAP.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=_model_device(model), dtype=model.cfg.dtype)    

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=attention_mask)
            elif 'mean' in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_tokens, attention_mask=attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor], steps=30, quiet=False):
    """Gets edge attribution scores using EAP with integrated gradients.

    Args:
        model (HookedTransformer): The model to attribute
        graph (Graph): Graph to attribute
        dataloader (DataLoader): The data over which to attribute
        metric (Callable[[Tensor], Tensor]): metric to attribute with respect to
        steps (int, optional): number of IG steps. Defaults to 30.
        quiet (bool, optional): suppress tqdm output. Defaults to False.

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=_model_device(model), dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, n_pos_corrupted = tokenize_plus(model, corrupted)

        if n_pos != n_pos_corrupted:
            print(f"Number of positions must match, but do not: {n_pos} (clean) != {n_pos_corrupted} (corrupted)")
            print(clean)
            print(corrupted)
            raise ValueError("Number of positions must match")

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations 
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            input_activations_corrupted = activation_difference[:, :, graph.forward_index(graph.nodes['input'])].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)

            input_activations_clean = input_activations_corrupted - activation_difference[:, :, graph.forward_index(graph.nodes['input'])]

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (input_activations_clean - input_activations_corrupted) 
                new_input.requires_grad = True 
                return new_input
            return hook_fn

        total_steps = 0
        for step in range(0, steps):
            total_steps += 1
            with model.hooks(fwd_hooks=[(graph.nodes['input'].out_hook, input_interpolation_hook(step))], bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                if torch.isnan(metric_value).any().item():
                    print("Metric value is NaN")
                    print(f"Clean: {clean}")
                    print(f"Corrupted: {corrupted}")
                    print(f"Label: {label}")
                    print(f"Metric: {metric}")
                    raise ValueError("Metric value is NaN")
                metric_value.backward()
            
            if torch.isnan(scores).any().item():
                print("Metric value is NaN")
                print(f"Clean: {clean}")
                print(f"Corrupted: {corrupted}")
                print(f"Label: {label}")
                print(f"Metric: {metric}")
                print(f'Step: {step}')
                raise ValueError("Metric value is NaN")

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_ig_activations(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
                              metric: Callable[[Tensor], Tensor], intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                              steps=30, intervention_dataloader: Optional[DataLoader]=None, quiet=False):

    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=_model_device(model), dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        if intervention == 'patching':
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

        elif 'mean' in intervention:
            activation_difference += means


        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=attention_mask)
            activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output
            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in nodeslist:
            for step in range(1, steps+1):
                total_steps += 1
                
                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_clean_corrupted(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
                               metric: Callable[[Tensor], Tensor], quiet=False):
    """Gets scores using the clean-corrupted method: like EAP-IG, but just do it on the clean and corrupted inputs, instead of all the intermediate steps.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        _type_: _description_
    """

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=_model_device(model), dtype=model.cfg.dtype)    
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=attention_mask)


        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

            corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            corrupted_metric_value = metric(corrupted_logits, clean_logits, input_lengths, label)
            corrupted_metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores

def get_scores_information_flow_routes(model: HookedTransformer, graph: Graph, dataloader: DataLoader, quiet=False) -> torch.Tensor:
    """Gets scores using Ferrando et al.'s (2024) information flow routes method.

    Args:
        model (HookedTransformer): the model to attribute
        graph (Graph): the graph to attribute
        dataloader (DataLoader): the data over which to attribute
        metric (Callable[[Tensor], Tensor]): the metric to attribute with respect to
        quiet (bool, optional): whether to silence tqdm. Defaults to False.

    Returns:
        Tensor: scores based on information flow routes
    """
    # I could do some hacky overriding of make_hooks_and_matrices here but I will not
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=_model_device(model), dtype=model.cfg.dtype)    

    def make_hooks(n_pos: int, input_lengths: torch.Tensor) -> List[Tuple[str, Callable]]:
        output_activations = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model), device=model.cfg.device, dtype=model.cfg.dtype)

        def output_hook(index, activations, hook):
            try:
                acts = activations.detach()
                output_activations[:, :, index] = acts
            except RuntimeError as e:
                print(hook.name, output_activations[:, :, index].size(), output_activations.size())
                raise e

        # compute the score directly, without saving the input activations
        def input_hook(prev_index, bwd_index, input_lengths, activations, hook):
            acts = activations.detach()
            try:
                if acts.ndim == 3:
                    acts = acts.unsqueeze(2)
                # acts : batch pos backward hidden
                # output acts: batch pos forward hidden
                # add forward and backwards dimensions to acts and output acts respectively
                acts = acts.unsqueeze(2)
                unsqueezed_output_activations = output_activations.unsqueeze(3)

                # acts : batch pos 1 backward hidden
                # output acts: batch pos forward 1 hidden
                proximity = torch.clamp(- torch.linalg.vector_norm(unsqueezed_output_activations[:, :, :prev_index] - acts, ord=1, dim=-1) + torch.linalg.vector_norm(acts, ord=1, dim=-1), min=0)
                importance = proximity / torch.sum(proximity, dim=2, keepdim=True)
                # importance: batch pos forward backward
                # aggregate over positions via sum/mean to get importance: forward backward
                # first mask out importances for padding positions
                max_len = input_lengths.max()
                mask = torch.arange(max_len, device=input_lengths.device,
                            dtype=input_lengths.dtype).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                # print(importance.size(), mask.size())
                importance *= mask
                importance = importance.sum(1) / input_lengths.view(-1,1,1) # mean over positions
                importance = importance.sum(0)

                # importance: forward backward
                # squeezing backward dim in case it isn't real (i.e. it's an MLP)
                importance = importance.squeeze(1)
                scores[:prev_index, bwd_index] += importance

            except RuntimeError as e:
                print(hook.name, unsqueezed_output_activations[:, :, prev_index].size(), acts.size())
                raise e
            
        hooks = []
        node = graph.nodes['input']
        fwd_index = graph.forward_index(node)
        hooks.append((node.out_hook, partial(output_hook, fwd_index)))
        
        for layer in range(graph.cfg['n_layers']):
            node = graph.nodes[f'a{layer}.h0']
            fwd_index = graph.forward_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            prev_index = graph.prev_index(node)
            for i, letter in enumerate('qkv'):
                bwd_index = graph.backward_index(node, qkv=letter)
                hooks.append((node.qkv_inputs[i], partial(input_hook, prev_index, bwd_index, input_lengths)))

            node = graph.nodes[f'm{layer}']
            fwd_index = graph.forward_index(node)
            bwd_index = graph.backward_index(node)
            prev_index = graph.prev_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
            
        node = graph.nodes['logits']
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
        return hooks
    
    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, _, _ in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)

        hooks = make_hooks(n_pos, input_lengths)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=hooks):
                _ = model(clean_tokens, attention_mask=attention_mask)

    scores /= total_items

    return scores

def get_scores_gwai(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                    metric: Callable[[Tensor], Tensor],
                    scoring: Literal['combined', 'gradient', 'proximity'] = 'combined',
                    incremental: bool = True,
                    tsg_temperature: float = 2.0, scale_multiplicative: bool = True,
                    chunk_size: int = 8, quiet: bool = False) -> torch.Tensor:
    """Gets scores using GWAI: Gradient-Weighted ALTI Interactions.

    Uses counterfactual activation differences (corrupted - clean) for gradient-
    based scoring, and clean source contributions for proximity weighting.
    When incremental=True, activation differences are propagated through each
    layer via GIM-corrected JVPs, accounting for self-repair effects.

    Three scoring modes:
      - 'combined' (default): proximity-weighted counterfactual gradient projection.
        score_j = proximity(z_j^clean, y^clean) × ((z_j^corr - z_j^clean) · grad).
        Edges must be both structurally important (ALTI) AND task-relevant (gradient).
      - 'gradient': counterfactual gradient projection ((z_j^corr - z_j^clean) · grad).
      - 'proximity': ALTI proximity metric (geometric, task-agnostic, no corrupted pass).

    When incremental=False, no JVP propagation is done. For proximity-only mode
    this gives standard ALTI. For gradient/combined modes, raw activation diffs
    are used without propagation.

    Args:
        model: the model to attribute
        graph: the graph to attribute
        dataloader: the data over which to attribute
        metric: the metric to attribute w.r.t. (needed for combined/gradient scoring)
        scoring: 'combined', 'gradient', or 'proximity'
        incremental: whether to propagate sources through layers via GIM JVPs
        tsg_temperature: temperature for TSG softmax correction (default 2.0)
        scale_multiplicative: whether to apply Shapley /2 at multiplicative junctions
        chunk_size: number of source nodes to process simultaneously in JVP
        quiet: suppress tqdm output

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    device = _model_device(model)
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)

    needs_grad = scoring in ('combined', 'gradient')
    needs_diff = needs_grad  # activation diffs needed for gradient-based scoring
    names_filter = make_names_filter(model, needs_jvp=incremental)
    n_layers = graph.cfg['n_layers']
    n_heads = graph.cfg['n_heads']
    parallel = model.cfg.parallel_attn_mlp

    total_items = 0
    dataloader_iter = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader_iter:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)

        # --- Clean forward pass (full cache for JVPs and proximity) ---
        with torch.inference_mode():
            _, cache = model.run_with_cache(clean_tokens, attention_mask=attention_mask,
                                            names_filter=names_filter)

        # --- Corrupted forward pass (source outputs only) ---
        corrupted_cache = None
        if needs_diff:
            corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)
            _src_hooks = {'hook_embed'}
            for _l in range(n_layers):
                _src_hooks.add(f'blocks.{_l}.attn.hook_result')
                _src_hooks.add(f'blocks.{_l}.hook_mlp_out')
            with torch.inference_mode():
                _, corrupted_cache = model.run_with_cache(
                    corrupted_tokens, attention_mask=attention_mask,
                    names_filter=lambda name: name in _src_hooks)

        # --- Compute task gradients at each destination input ---
        grad_at = {}  # key -> gradient tensor
        if needs_grad:
            saved_acts = {}
            def make_save_hook(key):
                def hook_fn(activations, hook):
                    activations.retain_grad()
                    saved_acts[key] = activations
                    return activations
                return hook_fn

            fwd_hooks = []
            for layer in range(n_layers):
                attn_node = graph.nodes[f'a{layer}.h0']
                for i, letter in enumerate('qkv'):
                    hook_name = attn_node.qkv_inputs[i]
                    fwd_hooks.append((hook_name, make_save_hook((layer, letter))))
                mlp_node = graph.nodes[f'm{layer}']
                fwd_hooks.append((mlp_node.in_hook, make_save_hook((layer, 'mlp'))))
            logit_node = graph.nodes['logits']
            fwd_hooks.append((logit_node.in_hook, make_save_hook('logits')))

            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                clean_logits = logits.detach()
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

            for key, act in saved_acts.items():
                if act.grad is not None:
                    grad_at[key] = act.grad.detach()
            model.zero_grad()

        # --- Build source activation tensors ---
        def _build_source_acts(src_cache):
            sa = torch.zeros((batch_size, n_pos, graph.n_forward, model.cfg.d_model),
                             device=device, dtype=model.cfg.dtype)
            sa[:, :, 0] = src_cache['hook_embed']
            for _l in range(n_layers):
                _an = graph.nodes[f'a{_l}.h0']
                _fi = graph.forward_index(_an)
                sa[:, :, _fi] = src_cache[f'blocks.{_l}.attn.hook_result']
                _mn = graph.nodes[f'm{_l}']
                _mi = graph.forward_index(_mn, attn_slice=False)
                sa[:, :, _mi] = src_cache[f'blocks.{_l}.hook_mlp_out']
            return sa

        # Clean source contributions (for proximity weighting)
        source_acts_clean = _build_source_acts(cache)

        # Activation differences: corrupted - clean (for gradient scoring)
        source_acts_diff = None
        if needs_diff:
            source_acts_diff = _build_source_acts(corrupted_cache) - source_acts_clean
            del corrupted_cache

        # --- Incremental scoring ---
        def _score_attn_dest(layer, letter, prev_idx):
            """Score sources -> attention Q/K/V destination."""
            attn_n = graph.nodes[f'a{layer}.h0']
            key = (layer, letter)
            bwd_idx = graph.backward_index(attn_n, qkv=letter)
            if scoring == 'combined' and key in grad_at:
                ref = cache[f'blocks.{layer}.hook_resid_pre']
                edge_scores = compute_combined_scores(
                    source_acts_clean[:, :, :prev_idx],
                    source_acts_diff[:, :, :prev_idx],
                    ref, grad_at[key], input_lengths)
            elif scoring == 'gradient' and key in grad_at:
                edge_scores = compute_edge_gradient_scores(
                    source_acts_diff[:, :, :prev_idx], grad_at[key], input_lengths)
            else:
                ref = cache[f'blocks.{layer}.hook_resid_pre']
                importance = compute_proximity_scores(
                    source_acts_clean[:, :, :prev_idx], ref, input_lengths)
                scores[:prev_idx, bwd_idx] += importance.unsqueeze(1).expand(-1, n_heads)
                return
            if edge_scores.ndim == 1:
                scores[:prev_idx, bwd_idx] += edge_scores.unsqueeze(1).expand(-1, n_heads)
            else:
                scores[:prev_idx, bwd_idx] += edge_scores

        def _score_mlp_dest(layer, prev_idx):
            """Score sources -> MLP destination."""
            mlp_n = graph.nodes[f'm{layer}']
            bwd_idx = graph.backward_index(mlp_n)
            if scoring == 'combined' and (layer, 'mlp') in grad_at:
                ref = (cache[f'blocks.{layer}.hook_resid_mid'] if not parallel
                       else cache[f'blocks.{layer}.hook_resid_pre'])
                edge_scores = compute_combined_scores(
                    source_acts_clean[:, :, :prev_idx],
                    source_acts_diff[:, :, :prev_idx],
                    ref, grad_at[(layer, 'mlp')], input_lengths)
            elif scoring == 'gradient' and (layer, 'mlp') in grad_at:
                edge_scores = compute_edge_gradient_scores(
                    source_acts_diff[:, :, :prev_idx], grad_at[(layer, 'mlp')], input_lengths)
            else:
                ref = (cache[f'blocks.{layer}.hook_resid_mid'] if not parallel
                       else cache[f'blocks.{layer}.hook_resid_pre'])
                edge_scores = compute_proximity_scores(
                    source_acts_clean[:, :, :prev_idx], ref, input_lengths)
            scores[:prev_idx, bwd_idx] += edge_scores

        for layer in range(n_layers):
            attn_node = graph.nodes[f'a{layer}.h0']
            prev_index = graph.prev_index(attn_node)

            if prev_index > 0:
                for letter in 'qkv':
                    _score_attn_dest(layer, letter, prev_index)

            # Propagate through attention of this layer
            if incremental:
                n_src_before_attn = graph.prev_index(attn_node)
                if n_src_before_attn > 0:
                    with torch.inference_mode():
                        if needs_diff:
                            source_acts_diff = _propagate_chunked_attention(
                                source_acts_diff, n_src_before_attn, layer, model, cache,
                                tsg_temperature, scale_multiplicative, chunk_size)
                        else:
                            source_acts_clean = _propagate_chunked_attention(
                                source_acts_clean, n_src_before_attn, layer, model, cache,
                                tsg_temperature, scale_multiplicative, chunk_size)

            # --- MLP destination ---
            mlp_node = graph.nodes[f'm{layer}']
            prev_index_mlp = graph.prev_index(mlp_node)

            if prev_index_mlp > 0:
                _score_mlp_dest(layer, prev_index_mlp)

            # Propagate through MLP of this layer
            if incremental:
                n_src_before_mlp = graph.prev_index(mlp_node)
                if n_src_before_mlp > 0:
                    with torch.inference_mode():
                        if needs_diff:
                            source_acts_diff = _propagate_chunked_mlp(
                                source_acts_diff, n_src_before_mlp, layer, model, cache,
                                scale_multiplicative, chunk_size)
                        else:
                            source_acts_clean = _propagate_chunked_mlp(
                                source_acts_clean, n_src_before_mlp, layer, model, cache,
                                scale_multiplicative, chunk_size)

        # --- Logits destination ---
        logit_node = graph.nodes['logits']
        prev_index_logits = graph.prev_index(logit_node)

        if scoring == 'combined' and 'logits' in grad_at:
            ref_logits = cache[f'blocks.{n_layers - 1}.hook_resid_post']
            importance_logits = compute_combined_scores(
                source_acts_clean[:, :, :prev_index_logits],
                source_acts_diff[:, :, :prev_index_logits],
                ref_logits, grad_at['logits'], input_lengths)
        elif scoring == 'gradient' and 'logits' in grad_at:
            importance_logits = compute_edge_gradient_scores(
                source_acts_diff[:, :, :prev_index_logits], grad_at['logits'], input_lengths)
        else:
            ref_logits = cache[f'blocks.{n_layers - 1}.hook_resid_post']
            importance_logits = compute_proximity_scores(
                source_acts_clean[:, :, :prev_index_logits], ref_logits, input_lengths)

        bwd_idx_logits = graph.backward_index(logit_node)
        scores[:prev_index_logits, bwd_idx_logits] += importance_logits

        del cache, source_acts_clean
        if source_acts_diff is not None:
            del source_acts_diff
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores /= total_items
    return scores


def get_scores_gim(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                    metric: Callable[[Tensor], Tensor],
                    intervention: Literal['patching', 'zero', 'mean', 'mean-positional'] = 'patching',
                    intervention_dataloader: Optional[DataLoader] = None,
                    quiet: bool = False) -> torch.Tensor:
    """Gets edge attribution scores using GIM (Gradient-based Interpretability Method).

    Uses the gim-explain package's context manager to patch the model's backward
    pass with GIM's modified gradient rules:
    - Frozen LayerNorm (detach normalization statistics)
    - Temperature-scaled softmax gradient (TSG, T=2.0)
    - Shapley normalization (Q/K ÷4, V ÷2)

    Edge scores are activation_diff × GIM-corrected gradient, computed via
    EAP's hook infrastructure within the GIM context.

    Args:
        model: the model to attribute
        graph: the graph to attribute
        dataloader: the data over which to attribute
        metric: the metric to attribute w.r.t.
        intervention: intervention type (same as EAP)
        intervention_dataloader: dataloader for mean interventions
        quiet: suppress tqdm output

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    import gim

    with gim.GIM(model):
        return get_scores_eap(model, graph, dataloader, metric,
                              intervention=intervention,
                              intervention_dataloader=intervention_dataloader,
                              quiet=quiet)


allowed_aggregations = {'sum', 'mean'}
def attribute(model: HookedTransformer, graph: Graph, dataloader: DataLoader, metric: Callable[[Tensor], Tensor],
              method: Literal['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'GWAI', 'GIM', 'exact'],
              intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', aggregation='sum',
              ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, quiet=False,
              gwai_scoring: str = 'combined', gwai_incremental: bool = True,
              gwai_tsg_temperature: float = 2.0, gwai_scale_multiplicative: bool = True,
              gwai_chunk_size: int = 8):
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    assert model.cfg.use_split_qkv_input, "Model must be configured to use split qkv inputs (model.cfg.use_split_qkv_input)"
    assert model.cfg.use_hook_mlp_in, "Model must be configured to use hook MLP in (model.cfg.use_hook_mlp_in)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
    
    if aggregation not in allowed_aggregations:
        raise ValueError(f'aggregation must be in {allowed_aggregations}, but got {aggregation}')
        
    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == 'EAP':
        scores = get_scores_eap(model, graph, dataloader, metric, intervention=intervention, 
                                intervention_dataloader=intervention_dataloader, quiet=quiet)
    elif method == 'EAP-IG-inputs':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}")
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == 'clean-corrupted':
        if intervention != 'patching':
            raise ValueError(f"intervention must be 'patching' for clean-corrupted, but got {intervention}")
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet)
    elif method == 'EAP-IG-activations':
        scores = get_scores_ig_activations(model, graph, dataloader, metric, steps=ig_steps, intervention=intervention, 
                                           intervention_dataloader=intervention_dataloader, quiet=quiet)
    elif method == 'information-flow-routes':
        scores = get_scores_information_flow_routes(model, graph, dataloader, quiet=quiet)
    elif method == 'GWAI':
        scores = get_scores_gwai(model, graph, dataloader, metric=metric, scoring=gwai_scoring,
                                 incremental=gwai_incremental, tsg_temperature=gwai_tsg_temperature,
                                 scale_multiplicative=gwai_scale_multiplicative,
                                 chunk_size=gwai_chunk_size, quiet=quiet)
    elif method == 'GIM':
        scores = get_scores_gim(model, graph, dataloader, metric, quiet=quiet)
    elif method == 'exact':
        scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention, intervention_dataloader=intervention_dataloader,
                                  quiet=quiet)
    else:
        raise ValueError(f"method must be in ['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'GWAI', 'GIM', 'exact'], but got {method}")


    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    graph.scores[:] =  scores.to(graph.scores.device)


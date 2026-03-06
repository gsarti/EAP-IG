from typing import Callable, List, Optional, Literal, Tuple
from functools import partial
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer

from tqdm import tqdm

from .utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations
from .evaluate import evaluate_graph, evaluate_baseline
from .graph import Graph
from .pf_gim import compute_proximity_scores


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

def get_scores_pf_gim(model: HookedTransformer, graph: Graph, dataloader: DataLoader,
                      metric: Callable[[Tensor], Tensor],
                      use_gim_grad: bool = True,
                      filter_quantile: float = 0.35,
                      quiet: bool = False) -> torch.Tensor:
    """Gets scores using PF-GIM: Proximity-Filtered GIM.

    Computes GIM-corrected gradient scores (activation_diff × GIM_grad) for all
    edges, then filters out structurally implausible edges using ALTI proximity.
    Edges with proximity below a quantile threshold are zeroed out. Gradient
    provides ranking, proximity provides structural plausibility filtering.

    Uses a 2-pass hook-based approach:
      Pass 1 (corrupted, inference mode): fills activation differences via hooks
      Pass 2 (clean, forward+backward): computes proximity in forward hooks,
        gradient scores in backward hooks

    Args:
        model: the model to attribute
        graph: the graph to attribute
        dataloader: the data over which to attribute
        metric: the metric to attribute w.r.t.
        use_gim_grad: use GIM-corrected gradients (frozen LN, TSG softmax, Shapley)
        filter_quantile: quantile threshold for proximity filtering (default 0.35)
        quiet: suppress tqdm output

    Returns:
        Tensor: a [src_nodes, dst_nodes] tensor of scores for each edge
    """
    device = _model_device(model)
    n_layers = graph.cfg['n_layers']
    n_heads = graph.cfg['n_heads']
    parallel = model.cfg.parallel_attn_mlp

    scores_grad = torch.zeros((graph.n_forward, graph.n_backward), device=device, dtype=model.cfg.dtype)
    scores_prox = torch.zeros_like(scores_grad)

    total_items = 0
    dataloader_iter = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader_iter:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        # Shared tensors filled by hooks
        activation_difference = torch.zeros(
            (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
            device=device, dtype=model.cfg.dtype)
        source_acts_clean = torch.zeros_like(activation_difference)

        # Position mask for input_lengths masking in backward hooks
        position_mask = (torch.arange(n_pos, device=device).expand(batch_size, n_pos)
                         < input_lengths.unsqueeze(1))

        # --- Hook definitions ---
        def corrupted_source_hook(fwd_index, activations, hook):
            activation_difference[:, :, fwd_index] += activations.detach()

        def clean_source_hook(fwd_index, activations, hook):
            acts = activations.detach()
            activation_difference[:, :, fwd_index] -= acts
            source_acts_clean[:, :, fwd_index] = acts

        def dest_fwd_hook(prev_index, bwd_index, is_attn, activations, hook):
            """Compute proximity scores during clean forward pass."""
            ref = activations.detach()
            if ref.ndim == 4:  # split QKV: (batch, pos, n_heads, d_model)
                ref = ref[:, :, 0, :]
            ep = compute_proximity_scores(
                source_acts_clean[:, :, :prev_index], ref, input_lengths)
            if is_attn:
                scores_prox[:prev_index, bwd_index] += ep.unsqueeze(1).expand_as(
                    scores_prox[:prev_index, bwd_index])
            else:
                scores_prox[:prev_index, bwd_index] += ep

        def dest_bwd_hook(prev_index, bwd_index, gradients, hook):
            """Compute gradient scores during backward pass."""
            grads = gradients.detach()
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            # Mask gradients to zero out padding positions
            masked_grads = grads * position_mask.unsqueeze(-1).unsqueeze(-1)
            s = torch.einsum('bpfh,bpkh->fk',
                             activation_difference[:, :, :prev_index], masked_grads)
            s = s.squeeze(-1)
            scores_grad[:prev_index, bwd_index] += s

        # --- Build corrupted hooks (source outputs only) ---
        fwd_hooks_corrupted = []
        node = graph.nodes['input']
        fwd_hooks_corrupted.append((node.out_hook,
                                    partial(corrupted_source_hook, graph.forward_index(node))))
        for layer in range(n_layers):
            node = graph.nodes[f'a{layer}.h0']
            fwd_hooks_corrupted.append((node.out_hook,
                                        partial(corrupted_source_hook, graph.forward_index(node))))
            node = graph.nodes[f'm{layer}']
            fwd_hooks_corrupted.append((node.out_hook,
                                        partial(corrupted_source_hook, graph.forward_index(node, attn_slice=False))))

        # --- Build clean forward + backward hooks ---
        fwd_hooks_clean = []
        bwd_hooks = []

        # Embed source hook
        node = graph.nodes['input']
        fwd_hooks_clean.append((node.out_hook,
                                partial(clean_source_hook, graph.forward_index(node))))

        for layer in range(n_layers):
            attn_node = graph.nodes[f'a{layer}.h0']
            prev_index_attn = graph.prev_index(attn_node)

            # Attention destination hooks (Q, K, V) — fire before attn source output
            if prev_index_attn > 0:
                for i, letter in enumerate('qkv'):
                    bwd_index = graph.backward_index(attn_node, qkv=letter)
                    fwd_hooks_clean.append((attn_node.qkv_inputs[i],
                                            partial(dest_fwd_hook, prev_index_attn, bwd_index, True)))
                    bwd_hooks.append((attn_node.qkv_inputs[i],
                                      partial(dest_bwd_hook, prev_index_attn, bwd_index)))

            # Attention source output hook
            fwd_hooks_clean.append((attn_node.out_hook,
                                    partial(clean_source_hook, graph.forward_index(attn_node))))

            mlp_node = graph.nodes[f'm{layer}']
            prev_index_mlp = graph.prev_index(mlp_node)
            bwd_index_mlp = graph.backward_index(mlp_node)

            # MLP destination hooks — fire after attn source, before MLP source
            if prev_index_mlp > 0:
                fwd_hooks_clean.append((mlp_node.in_hook,
                                        partial(dest_fwd_hook, prev_index_mlp, bwd_index_mlp, False)))
                bwd_hooks.append((mlp_node.in_hook,
                                  partial(dest_bwd_hook, prev_index_mlp, bwd_index_mlp)))

            # MLP source output hook
            fwd_hooks_clean.append((mlp_node.out_hook,
                                    partial(clean_source_hook, graph.forward_index(mlp_node, attn_slice=False))))

        # Logits destination hooks
        logit_node = graph.nodes['logits']
        prev_index_logits = graph.prev_index(logit_node)
        bwd_index_logits = graph.backward_index(logit_node)
        fwd_hooks_clean.append((logit_node.in_hook,
                                partial(dest_fwd_hook, prev_index_logits, bwd_index_logits, False)))
        bwd_hooks.append((logit_node.in_hook,
                          partial(dest_bwd_hook, prev_index_logits, bwd_index_logits)))

        # --- Pass 1: Corrupted forward (fill activation_difference) ---
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                model(corrupted_tokens, attention_mask=attention_mask)

        # --- Pass 2: Clean forward (source acts + proximity) + backward (gradient scores) ---
        gim_ctx = nullcontext()
        if use_gim_grad:
            import gim
            gim_ctx = gim.GIM(model)

        with gim_ctx:
            with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
                logits = model(clean_tokens, attention_mask=attention_mask)
                clean_logits = logits.detach()
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

        model.zero_grad()
        del activation_difference, source_acts_clean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Proximity-filtered gradient scores ---
    scores_grad /= total_items
    scores_prox /= total_items

    prox_flat = scores_prox[scores_prox > 0]
    if prox_flat.numel() > 0:
        threshold = torch.quantile(prox_flat, filter_quantile)
        mask = scores_prox >= threshold
        return scores_grad * mask
    return scores_grad


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
              method: Literal['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'PF-GIM', 'GIM', 'exact'],
              intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', aggregation='sum',
              ig_steps: Optional[int]=None, intervention_dataloader: Optional[DataLoader]=None, quiet=False,
              pf_gim_use_gim_grad: bool = True, pf_gim_filter_quantile: float = 0.35):
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
    elif method == 'PF-GIM':
        scores = get_scores_pf_gim(model, graph, dataloader, metric=metric,
                                   use_gim_grad=pf_gim_use_gim_grad,
                                   filter_quantile=pf_gim_filter_quantile,
                                   quiet=quiet)
    elif method == 'GIM':
        scores = get_scores_gim(model, graph, dataloader, metric, quiet=quiet)
    elif method == 'exact':
        scores = get_scores_exact(model, graph, dataloader, metric, intervention=intervention, intervention_dataloader=intervention_dataloader,
                                  quiet=quiet)
    else:
        raise ValueError(f"method must be in ['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'PF-GIM', 'GIM', 'exact'], but got {method}")


    if aggregation == 'mean':
        scores /= model.cfg.d_model
        
    graph.scores[:] =  scores.to(graph.scores.device)


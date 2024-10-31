import re
import time
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.fx as fx
from ..graph import (
    prune_and_merge_ops,
    remove_unimportant_nodes_and_reconnect,
    patch_grouped_linear,
    connect_graphs,
    remove_node,
    connect,
    get_encoder_leaf_nodes,
)
from ..utils import get_logger


def trace_model(model, bsz: int = 1, concrete_args: dict = {}):
    """
    Traces the model for one sequence and replicates
    the graph in the horizontal dimension for bsz many
    times.

    Args:
        encoder (TransformerEncoder): The encoder.
        bsz (int): The number of sequences/batch size (bsz).
    """
    assert bsz > 0, "bsz must be greater 0"
    connected_graph: fx.GraphModule = fx.symbolic_trace(
        model, concrete_args=concrete_args
    )
    current_seq_id = 0
    base_graph: fx.Graph = connected_graph.graph
    for node in base_graph.nodes:
        if "seq_" in node.name:
            # replace
            node.name = re.sub(r"seq_(\d+)", f"seq_{current_seq_id}", node.name)
        else:
            node.name = f"{node.name}_seq_{current_seq_id}"

    sequence_graphs = [deepcopy(base_graph) for _ in range(bsz - 1)]
    prev_graph_last_node = base_graph._root.prev
    for cloned_graph in sequence_graphs:
        base_graph._len += cloned_graph._len
        current_seq_id += 1
        for node in cloned_graph.nodes:
            node.name = re.sub(r"seq_(\d+)", f"seq_{current_seq_id}", node.name)
        prev_graph_last_node._next = cloned_graph._root._next
        prev_graph_last_node.op = "placeholder"
        prev_graph_last_node.name = f"output_seq_{current_seq_id}"
        prev_graph_last_node.users = {}
        for n in prev_graph_last_node._input_nodes:
            n.users.pop(prev_graph_last_node, None)
        prev_graph_last_node._input_nodes = {}
        prev_graph_last_node = cloned_graph._root._prev

    # wrap around. last output node connects to first root node again
    prev_graph_last_node._next = base_graph._root

    # finally, remove the placeholder
    for node in connected_graph.graph.nodes:
        if node.op == "placeholder" and "output" in node.name:
            remove_node(connected_graph.graph, node)
            for inp_node in node._input_nodes:
                inp_node.users.pop(node, None)
            node._input_nodes = {}

    return connected_graph


def trace_prune_and_patch(
    model: torch.nn.Module, seq_length_decoding_id: tuple, kv_caching: bool = False
):
    if len(seq_length_decoding_id) == 3:
        seq_length, decoding_id, context_len = seq_length_decoding_id
        concrete_args = {
            "context_len": context_len,
            "gen_start_len": seq_length,
            "gen_target_len": seq_length + 1,
        }
    else:
        assert (
            len(seq_length_decoding_id) == 2
        ), "Arguments for decoder-only must be length 2."
        seq_length, decoding_id = seq_length_decoding_id
        concrete_args = {
            "start_len": seq_length,
            "target_len": seq_length + 1,
        }

    trace_logger = get_logger("Tracer")
    t0 = time.time()
    symb_traced = trace_model(model, bsz=1, concrete_args=concrete_args)
    trace_logger.debug(f"Finished tracing in {time.time() - t0:.4f}s")
    t0 = time.time()
    # prune the graph
    prune_and_merge_ops(symb_traced.graph)
    trace_logger.debug(f"Finished prune_and_merge in {time.time() - t0:.4f}s")
    t0 = time.time()
    remove_unimportant_nodes_and_reconnect(symb_traced.graph)
    trace_logger.debug(f"Finished remove_unimportant in {time.time() - t0:.4f}s")
    t0 = time.time()
    # patch the grouped linear nodes with per-tier sub-graphs
    patch_grouped_linear(
        symb_traced.graph,
        tier_shape=model.accelerator.config.tier_shape,
        decoding_id=decoding_id,
    )
    # the patched linears introduced some new unimportant nodes that need to be removed
    remove_unimportant_nodes_and_reconnect(symb_traced.graph, kv_caching=kv_caching)
    trace_logger.debug(
        f"Finished patch & remove_unimportant in {time.time() - t0:.4f}s"
    )
    return symb_traced


def fast_trace_encoder_decoder(
    model: torch.nn.Module,
    context_len: int,
    gen_start_len: int,
    gen_target_len: int,
    bsz: int = 1,
):
    """
    Accelerates tracing for the autoregressive forward pass.
    Traces the models individually for decoding the next token
    for a given gen_start_len and then connects them. Then,
    the graph is replicated to simulate the execution on
    multiple sequences. In this case, we also prune away
    the encoder graphs that are replicated since we create individual
    graphs for each step.

    Args:
        model (torch.nn.Module): The model to trace.
        context_len (int): Sequence length of the context that is fed into the encoder.
        gen_start_len (int): How many tokens have we decoded so far?
        gen_target_len (int): How many tokens do we want to decode in total?
        bsz (int, optional): Number of sequences. Defaults to 1.
    """
    individual_graphs = []

    for decoding_id, seq_length in enumerate(range(gen_start_len, gen_target_len)):
        symb_traced = trace_prune_and_patch(
            model=model,
            seq_length_decoding_id=(seq_length, decoding_id, context_len),
        )
        individual_graphs.append(symb_traced)

    if len(individual_graphs) == 1:
        connected_graph = individual_graphs[0]
    else:
        connected_graph = individual_graphs[0]
        encoder_leaf_nodes, _ = get_encoder_leaf_nodes(connected_graph.graph)
        for g2 in individual_graphs[1:]:
            g2_encoder_leaf_nodes, g2_encoder_nodes = get_encoder_leaf_nodes(g2.graph)
            # all-to-all connection between the base leaf nodes and the users of the
            # g2_encoder_leaf_nodes users
            for u in encoder_leaf_nodes:
                for v in list(set([u for n in g2_encoder_leaf_nodes for u in n.users])):
                    connect(u, v)
            # remove the other encoder nodes from g2
            for u in g2_encoder_nodes:
                remove_node(g2.graph, u)

            connect_graphs(connected_graph.graph, g2.graph)

    # copy for multiple sequences
    base_graph: fx.Graph = connected_graph.graph
    current_seq_id = 0
    # all the nodes in the graph will have _seq_0
    for node in base_graph.nodes:
        if "seq_" in node.name:
            # replace
            node.name = re.sub(r"seq_(\d+)", f"seq_{current_seq_id}", node.name)
        else:
            node.name = f"{node.name}_seq_{current_seq_id}"
        node._args = tuple(node._input_nodes)
    sequence_graphs = [deepcopy(base_graph) for _ in tqdm(range(bsz - 1))]
    # if we add some (batched), we overwrite the seq_0 with the correct index

    prev_graph_last_node = base_graph._root.prev
    for cloned_graph in sequence_graphs:
        base_graph._len += cloned_graph._len
        current_seq_id += 1
        for node in cloned_graph.nodes:
            node.name = re.sub(r"seq_(\d+)", f"seq_{current_seq_id}", node.name)
        prev_graph_last_node._next = cloned_graph._root._next
        prev_graph_last_node.op = "placeholder"
        prev_graph_last_node.name = f"output_seq_{current_seq_id}"
        prev_graph_last_node.users = {}
        for n in prev_graph_last_node._input_nodes:
            n.users.pop(prev_graph_last_node, None)
        prev_graph_last_node._input_nodes = {}
        prev_graph_last_node = cloned_graph._root._prev

    # wrap around. last output node connects to first root node again
    prev_graph_last_node._next = base_graph._root

    # finally, remove the placeholder
    for node in connected_graph.graph.nodes:
        if node.op == "placeholder" and "output" in node.name:
            remove_node(connected_graph.graph, node)
            for inp_node in node._input_nodes:
                inp_node.users.pop(node, None)
            node._input_nodes = {}

    return connected_graph


def fast_trace_decoder(
    model: torch.nn.Module, start_len: int, target_len: int, bsz: int = 1
):
    """
    Accelerates tracing for the autoregressive forward pass.
    Traces the models individually for decoding the next token
    for a given start_len and then connects them. Then,
    the graph is replicated to simulate the execution on
    multiple sequences.

    Args:
        model (torch.nn.Module): The model to trace.
        start_len (int): Sequence length of the context.
        target_len (int): Sequence length to be reached.
        bsz (int, optional): Number of sequences. Defaults to 1.
    """
    individual_graphs = []

    # # Both not working :(
    # from multiprocessing import Pool
    # from functools import partial
    # from concurrent.futures import  ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=None) as e:
    #     individual_graphs = list(e.map(
    #         partial(trace_prune_and_patch, model), [*enumerate(range(start_len, target_len))]
    #     ))

    # # with Pool(8) as p:
    # #     individual_graphs = p.map(partial(trace_prune_and_patch, {"model": model}), [*enumerate(range(start_len, target_len))])
    kv_caching = model.accelerator.config.kv_caching
    symb_traced = trace_prune_and_patch(
        model=model,
        seq_length_decoding_id=(start_len, 0),
    )
    individual_graphs.append(symb_traced)
    for decoding_id, seq_length in enumerate(range(start_len + 1, target_len)):
        seq_length_decoding_id = 1 if kv_caching else seq_length
        symb_traced = trace_prune_and_patch(
            model=model,
            seq_length_decoding_id=(seq_length_decoding_id, decoding_id),
            kv_caching=kv_caching,
        )
        if kv_caching:
            # correct the seq len
            for node in symb_traced.graph.nodes:
                if "mha" in node.name:
                    node.kwargs["op_info"]["seq_len"] = seq_length

        individual_graphs.append(symb_traced)

    if len(individual_graphs) == 1:
        connected_graph = individual_graphs[0]
    else:
        connected_graph = individual_graphs[0]
        for g2 in individual_graphs[1:]:
            connect_graphs(connected_graph.graph, g2.graph)

    # copy for multiple sequences
    base_graph: fx.Graph = connected_graph.graph
    current_seq_id = 0
    # all the nodes in the graph will have _seq_0
    for node in base_graph.nodes:
        if "seq_" in node.name:
            # replace
            node.name = re.sub(r"seq_(\d+)", f"seq_{current_seq_id}", node.name)
        else:
            node.name = f"{node.name}_seq_{current_seq_id}"
        node._args = tuple(node._input_nodes)
    sequence_graphs = [deepcopy(base_graph) for _ in tqdm(range(bsz - 1))]
    # if we add some (batched), we overwrite the seq_0 with the correct index

    prev_graph_last_node = base_graph._root.prev
    for cloned_graph in sequence_graphs:
        base_graph._len += cloned_graph._len
        current_seq_id += 1
        for node in cloned_graph.nodes:
            node.name = re.sub(r"seq_(\d+)", f"seq_{current_seq_id}", node.name)
        prev_graph_last_node._next = cloned_graph._root._next
        prev_graph_last_node.op = "placeholder"
        prev_graph_last_node.name = f"output_seq_{current_seq_id}"
        prev_graph_last_node.users = {}
        for n in prev_graph_last_node._input_nodes:
            n.users.pop(prev_graph_last_node, None)
        prev_graph_last_node._input_nodes = {}
        prev_graph_last_node = cloned_graph._root._prev

    # wrap around. last output node connects to first root node again
    prev_graph_last_node._next = base_graph._root

    # finally, remove the placeholder
    for node in connected_graph.graph.nodes:
        if node.op == "placeholder" and "output" in node.name:
            remove_node(connected_graph.graph, node)
            for inp_node in node._input_nodes:
                inp_node.users.pop(node, None)
            node._input_nodes = {}

    return connected_graph

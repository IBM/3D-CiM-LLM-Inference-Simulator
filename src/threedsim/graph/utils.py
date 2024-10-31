import re
from tqdm import tqdm
import random
import string
from copy import deepcopy
import math
import torch
import torch.fx as fx
from torch.fx.graph import Graph
from torch.fx.node import Node

from ..utils import digital_add
from ..accelerator import Tier

fx.wrap(digital_add)

IMPORTANT_NODES = [
    "tier_linear",
    "grouped_linear",
    "mha",
    "layer_norm",
    "digital_relu",
    "digital_gelu",
    "digital_add",
    "communication",
    "multinomial",
]

FEATURE_OPS = [
    "tier_linear",
    "layer_norm",
    "digital_relu",
    "digital_gelu",
    "digital_add",
    "multinomial",
]


def get_random_str():
    characters = string.ascii_letters
    r_str = "".join(random.choice(characters) for _ in range(22))
    while "mha" in r_str:
        r_str = "".join(random.choice(characters) for _ in range(22))
    return r_str


def connect(u: Node, v: Node):
    """
    Connect u->v

    Args:
        u (Node): start node
        v (Node): end node
    """
    u.users[v] = None
    v._input_nodes[u] = None


def connect_important_nodes(transitive_closure: dict):
    """
    For every vertex, it it is an important vertex,
    connect it to the other reachable important vertices.

    Args:
        transitive_closure (dict): dict[Node, list[Node]]
    """
    start_node: Node
    for start_node in transitive_closure:
        if not is_important(start_node):
            continue
        reachable_node: Node
        for reachable_node in transitive_closure[start_node]:
            if not is_important(reachable_node):
                continue
            # connect
            connect(start_node, reachable_node)


def reachable_set(graph: Graph):
    """
    Computes the reachable set for every node
    in the graph.

    Args:
        graph (Graph): Graph to be used.

    Returns:
        (dict): Reachability relation v->[x,y,z]
    """
    reachability = {}
    node: Node
    for node in graph.nodes:
        reachable_set = dfs(node)
        reachability[node] = reachable_set
    return reachability


def dfs(start_node: Node):
    """
    Given start node, traverse the graph in
    depth-first manner and record the visited
    vertices in a list which is returned.

    Args:
        graph (Graph): _description_
        start_node (Node): _description_
    Returns:
        (list[Node]): Visited vertices (in-order)
    """
    stack = []
    traversed = {}
    stack.append(start_node)
    while len(stack) != 0:
        v: Node = stack.pop(-1)
        if not v in traversed:
            traversed[v] = None
            for u in v.users:
                stack.append(u)
    traversed = [*traversed]
    traversed.remove(start_node)
    return traversed


def remove_nodes(graph: Graph, nodes: list[Node]):
    """
    Removes a list of nodes from a graph in-place.

    Args:
        graph (Graph): Graph.
        nodes (list[Node]): List of nodes that are going to be removed.
    """
    for n in nodes:
        remove_node(graph, n)


def remove_node(graph: Graph, to_erase: Node):
    """
    Removes a node from the graph in-place.

    Args:
        graph (Graph): Graph to remove it from.
        to_erase (Node): Node to erase.
    """
    graph._len -= 1
    if len(to_erase.users) > 0:
        user_node: Node
        for user_node in to_erase.users:
            user_node._input_nodes.pop(to_erase, None)
            user_node._args = tuple([el for el in user_node._args if el != to_erase])

    to_erase._remove_from_list()
    to_erase._erased = True  # iterators may retain handles to erased nodes


def get_source_nodes(graph: Graph, assign_edge_degree: bool = False):
    """
    Retrieves the source nodes of a graph.
    A source node a is a node that doesn't
    have inputs.

    Args:
        graph (Graph): The graph.

    Returns:
        (list[Node]): List of source nodes.
    """
    source_nodes = []
    node: Node
    for node in graph.nodes:
        if node.op == "output":
            continue
        if node.all_input_nodes == [] and not node._erased:
            source_nodes.append(node)
        if assign_edge_degree:
            node.input_edge_degree = len(node._input_nodes)
    return source_nodes


def is_important(n: Node):
    """
    Check if a node is important (i.e. needs to be retained)

    Args:
        n (Node): Node to check.

    Returns:
        (bool): True if important.
    """
    important = False
    for imp_node in IMPORTANT_NODES:
        important = important or imp_node in n.name
    return important


def is_feature_node(n: Node):
    """
    Check if a node is a feature operation

    Args:
        n (Node): Node to check.

    Returns:
        (bool): True if feature operation.
    """
    feature = False
    for feat_op in FEATURE_OPS:
        feature = feature or feat_op in n.name
    return feature


def _is_digital(n: Node):
    return not ("tier_linear" in n.name)


def connect_graphs(g1: fx.Graph, g2: fx.Graph):
    """
    Connects graph g1 with graph g2.

    Args:
        g1 (fx.Graph): Base graph.
        g2 (fx.Graph): Graph to connect to base graph.
    """
    g1._len += g2._len
    names_g1 = {}
    leaf_nodes_g1 = []
    for node in g1.nodes:
        if node.op == "output":
            continue
        names_g1[node.name] = None
        true_users = [u for u in node.users if u.op != "output"]
        if len(true_users) == 0:
            leaf_nodes_g1.append(node)

    # we only want to keep the multinomial nodes
    leaf_nodes_g1 = [n for n in leaf_nodes_g1 if "multinomial" in n.name]

    source_nodes_g2 = []
    for node in g2.nodes:
        if names_g1.get(node.name, -1) is None:  # name is present
            node.name += f"_{get_random_str()}"
        if len(node._input_nodes) == 0:
            source_nodes_g2.append(node)

    # connect leaf_nodes with source nodes
    for u in leaf_nodes_g1:
        for v in source_nodes_g2:
            connect(u, v)

    # g1._root._prev is the last node of g2.
    last_node_g1 = g1._root._prev
    assert last_node_g1.op == "output", "The last node must be an output node"
    # change to placeholder op
    last_node_g1.op = "placeholder"

    last_node_g2 = g2._root._prev
    first_node_g2 = g2._root._next
    last_node_g1._next = first_node_g2
    first_node_g2._prev = last_node_g1
    g1._root._prev = last_node_g2
    last_node_g2._next = g1._root
    last_node_g2._args = tuple([*last_node_g2._input_nodes])


def patch_grouped_linear(graph: fx.Graph, tier_shape: tuple, decoding_id: int):
    """
    Goes through the graph and replaces linear calls with
    tiled linear calls using the actual mapping. The mapping
    is stored in the kwargs["op_info"] of the linear operation.

    Args:
        graph (fx.Graph): Graph to be patched.
        tier_shape (tuple): Shape of a single tier.
        decoding_id (int):
    """
    unique_shapes = list(
        set(
            [
                n.kwargs["op_info"]["shape"]
                for n in graph.nodes
                if "grouped_linear" in n.name
            ]
        )
    )
    mvm_graphs = {
        shape: create_base_graph_for_shape(weight_shape=shape, tier_shape=tier_shape)
        for shape in unique_shapes
    }

    node: Node
    for node in tqdm(graph.nodes, disable=False):
        if (
            decoding_id is not None
            and "op_info" in node.kwargs
            and "decoding_id" in node.kwargs["op_info"]
        ):
            new_kwargs = dict(node._kwargs)
            new_kwargs["op_info"] = {
                **new_kwargs["op_info"],
                "decoding_id": "memory"
                if node.kwargs["op_info"]["decoding_id"] == "memory"
                else decoding_id,
            }
            node._kwargs = new_kwargs

        if "grouped_linear" in node.name:
            mapping = node.kwargs["op_info"]["mapping"]
            mapping_unrolled = [el for h in mapping for el in h]
            mapping_offset = 0
            shape = node.kwargs["op_info"]["shape"]
            replace_graph: Graph = deepcopy(mvm_graphs[shape])
            graph._len += (
                replace_graph._len - 1
            )  # minus one because of the node we replace
            replace_node: Node
            for replace_node in replace_graph.nodes:
                # assign the new graph
                replace_node.graph = graph
                if replace_node.name == "dummy_token":
                    # this is the src node of the replace graph
                    src_replace_node = replace_node
                elif replace_node.name == "output":
                    # change this from output node to placeholder
                    # so it gets removed
                    replace_node.op = "placeholder"
                    sink_replace_node = replace_node
                elif "tier_linear" in replace_node.name:
                    tile_idx, tier_idx, utilization, n_rows, n_cols = mapping_unrolled[
                        mapping_offset
                    ]
                    mapping_offset += 1
                    new_kwargs = dict(replace_node._kwargs)
                    new_kwargs["op_info"] = {
                        "tile_idx": tile_idx,
                        "tier_idx": tier_idx,
                        "utilization": utilization,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        **node.kwargs["op_info"],
                        "decoding_id": "memory"
                        if node.kwargs["op_info"]["decoding_id"] == "memory"
                        else decoding_id,
                    }
                    new_kwargs["op_info"].pop("mapping")
                    replace_node._kwargs = new_kwargs
                    # If the tier linear layer has no number, add a zero
                    if re.search(r"\d+$", replace_node.name) is None:
                        replace_node.name += "_0"
                elif "digital_add" in replace_node.name:
                    new_kwargs = dict(replace_node._kwargs)
                    new_kwargs["op_info"] = {
                        **new_kwargs["op_info"],
                        **node.kwargs["op_info"],
                        "decoding_id": "memory"
                        if node.kwargs["op_info"]["decoding_id"] == "memory"
                        else decoding_id,
                    }
                    new_kwargs["op_info"].pop("mapping")
                    new_kwargs["op_info"].pop("shape")
                    replace_node._kwargs = new_kwargs
                replace_node.name += f"_{get_random_str()}"

            src_replace_node._prev = node._prev
            node._prev._next = src_replace_node
            sink_replace_node._next = node._next
            node._next._prev = sink_replace_node
            for input_node in node._input_nodes:
                input_node.users.pop(node)
                input_node.users[src_replace_node] = None
                src_replace_node._input_nodes[input_node] = None
                src_replace_node._args += (input_node,)
            for user_node in node.users:
                user_node._input_nodes.pop(node)
                user_node._args = tuple([el for el in user_node._args if el != node])
                user_node._input_nodes[sink_replace_node] = None
                user_node._args += (sink_replace_node,)
                sink_replace_node.users[user_node] = None


class TierMVM(torch.nn.Module):
    """
    Helper module for generating the graph for a tiled MVM.
    """

    def __init__(self, tier_shape: tuple, weight_shape: tuple):
        """
        Instantiate this module with the tier shape
        and the shape of the matrix. Tracing over the
        forward will yield a graph for the tiled MVM.

        Args:
            tier_shape (tuple): Shape of the tier.
            weight_shape (tuple): Shape of the weight.
        """
        super().__init__()
        self.tier_rows, self.tier_cols = tier_shape
        in_features, out_features = weight_shape
        self.n_vertical = math.ceil(in_features / self.tier_rows)
        self.n_horizontal = math.ceil(out_features / self.tier_cols)
        self.dummy_tier = Tier(tier_shape=tier_shape, device="meta")

    def forward(self, dummy_token):
        c_token = torch.split(dummy_token, split_size_or_sections=self.tier_rows)
        outputs = []
        for i in range(self.n_vertical):
            hor_output = []
            for j in range(self.n_horizontal):
                hor_output.append(self.dummy_tier(c_token[i]))
            outputs.append(hor_output)

        # Add all results in the Y axis if self.n_vertical > 1
        if self.n_vertical > 1:
            add_op_info = {"size": self.tier_cols}
            aggregated_output = []
            for j in range(self.n_horizontal):
                temp_res = digital_add(
                    outputs[0][j], outputs[1][j], op_info=add_op_info
                )
                for i in range(2, self.n_vertical):
                    temp_res = digital_add(temp_res, outputs[i][j], op_info=add_op_info)
                aggregated_output.append(temp_res)
        else:
            aggregated_output = outputs[0]

        # Concatenate all results in the X axis if self.n_horizontal > 1
        output = (
            torch.cat(aggregated_output)
            if self.n_horizontal > 1
            else aggregated_output[0]
        )
        return output


def create_base_graph_for_shape(weight_shape: tuple, tier_shape: tuple):
    """
    Create a dummy graph that represents the tiled MVM of a token
    for a given weight shape and input shape.

    Args:
        weight_shape (tuple): Shape of the weight. MVM is performed as x @ W.
        tier_shape (tuple): Shape of one tier.

    Returns:
        (fx.Graph): Returns the graph of one tiled MVM for a given shape.
    """
    dummy_mvm = TierMVM(tier_shape=tier_shape, weight_shape=weight_shape)
    dummy_symbolic_traced: fx.GraphModule = fx.symbolic_trace(dummy_mvm)
    return dummy_symbolic_traced.graph


def get_encoder_leaf_nodes(graph: fx.Graph):
    """
    For a graph, finds the nodes part of the encoder and returns the leaf nodes of that,
    which are the nodes that connect into non-encoder nodes.

    Args:
        graph (fx.Graph): Graph.

    Returns:
        tuple[list[Node],list[Node]]: first element is list of leaf nodes. 2nd is list of encoder nodes.
    """
    encoder_leaf_nodes, encoder_nodes = [], []
    node: fx.Node
    for node in graph.nodes:
        if (
            "op_info" in node.kwargs
            and node.kwargs["op_info"]["decoding_id"] == "memory"
        ):
            encoder_nodes.append(node)
            is_encoder_leaf = all(
                [
                    "memory" != u.kwargs["op_info"]["decoding_id"]
                    for u in node.users
                    if "decoding_id" in u.kwargs["op_info"]
                ]
            )
            if is_encoder_leaf:
                encoder_leaf_nodes.append(node)
    return encoder_leaf_nodes, encoder_nodes

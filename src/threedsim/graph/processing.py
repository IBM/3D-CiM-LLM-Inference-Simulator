import re

from torch.fx.graph import Graph
from torch.fx.node import Node

from ..utils import AddressGenerator
from .utils import (
    _is_digital,
    connect,
    get_source_nodes,
    is_feature_node,
    is_important,
    remove_node,
    remove_nodes,
)


def prune_and_merge_ops(graph: Graph):
    """
    1) Iteratively removes all un-important source nodes.
    un-important meaning no analog-linear and no multi-head
    attention. Done until there are no un-important source nodes
    anymore.
    2) Identified chains in the graph and fuses them. A chain
    is a list of nodes that are directly connected. Each node
    can only have one output (except for the last one).
    For simplicity, the first node in the chain turns into the
    fused node. This operation does not preserve semantic correctness,
    i.e. the graph is not runnable anymore (but could be theoretically).
    3) Connects the fused node correctly to the graph.

    Args:
        graph (Graph): Graph to be pruned in-place.
    """
    # recursively remove any source nodes that are
    # not dependent on any linear op
    source_nodes = [n for n in get_source_nodes(graph) if not is_important(n)]
    while source_nodes != []:
        remove_nodes(graph, source_nodes)
        source_nodes = [n for n in get_source_nodes(graph) if not is_important(n)]

    for node in graph.nodes:
        if "add_dependency" in node.name:
            node_users = [*node.users]
            assert len(node._args) == len(
                node_users
            ), "add_dependency node must have same number of inputs and outputs"
            for arg_idx, arg_node in enumerate(node._args):
                u = arg_node
                v = node_users[arg_idx]
                if u in node._input_nodes:
                    # form a connection from u -> v
                    connect(u, v)
            for u in node._input_nodes:
                u.users.pop(node)
            for v in node.users:
                v._input_nodes.pop(node)

            remove_node(graph, node)


def add_comm_nodes(graph: Graph):
    id_generator = AddressGenerator()
    node: Node
    for node in graph.nodes:
        if not _is_digital(node):
            if any(["multinomial" in child.name for child in node.users]):
                assert all(["multinomial" in child.name for child in node.users])
                continue
            # Insert one and only one comm node
            comm_node = Node(
                graph,
                f"communication_{id_generator.get_addr()}_seq_{seq_id_from_op(node)}",
                "placeholder",
                "no_target",
                args=tuple(),
                kwargs={},
            )

            comm_node._input_nodes[node] = None
            comm_node._prev = node
            comm_node._next = node.next
            for child in node.users:
                comm_node.users[child] = None

            node._next = comm_node
            node.users = {comm_node: None}
            graph._len += 1
            for child in comm_node.users:
                child._input_nodes.pop(node)
                child._input_nodes[comm_node] = None
                if child._prev == node:
                    child._prev = comm_node
        elif not "communication" in node.name:
            if "multinomial" in node.name:
                # If it's a multinomial add a SINGLE comm node just before it
                # That is because we are doing a streaming/hierarchical argmax
                comm_node = Node(
                    graph,
                    f"communication_{id_generator.get_addr()}_seq_{seq_id_from_op(node)}",
                    "placeholder",
                    "no_target",
                    args=tuple(),
                    kwargs={},
                )

                graph._len += 1

                # The "users" structure of the lm_head layers need to point to the comm_node
                # Also the lm_head layer that has as _next the multinomial node, needs to have
                # the comm node instead
                for inp_n in node._input_nodes:
                    inp_n.users[comm_node] = None
                    inp_n.users.pop(node)
                    if inp_n._next == node:
                        inp_n._next = comm_node
                        comm_node._prev = inp_n

                # The input nodes to the multinomial are now the input nodes to the comm node
                for inp_n in node._input_nodes:
                    comm_node._input_nodes[inp_n] = None

                # The output of the comm node is the multinomial node and the
                comm_node.users[node] = None
                comm_node._next = node
                node._input_nodes = {comm_node: None}
                node._prev = comm_node

            children_to_pop = []
            comm_nodes = []
            analog_children = [child for child in node.users if not _is_digital(child)]
            # if len(analog_children) == 3:
            #     print(node.name)
            if len(analog_children) == 0:
                continue
            for child in analog_children:
                comm_node = Node(
                    graph,
                    f"communication_{id_generator.get_addr()}_seq_{seq_id_from_op(node)}",
                    "placeholder",
                    "no_target",
                    args=tuple(),
                    kwargs={},
                )
                comm_node._input_nodes[node] = None
                comm_node.users = {child: None}

                children_to_pop.append(child)
                comm_nodes.append(comm_node)

                child._input_nodes.pop(node)
                child._input_nodes[comm_node] = None
                graph._len += 1

            for child in children_to_pop:
                node.users.pop(child)
            for comm_node in comm_nodes:
                node.users[comm_node] = None

            # Patch prev/next
            for i in range(1, len(comm_nodes)):
                comm_nodes[i - 1]._next = comm_nodes[i]
                comm_nodes[i]._prev = comm_nodes[i - 1]

            comm_nodes[0]._prev = node
            comm_nodes[-1]._next = node._next
            node._next._prev = comm_nodes[-1]
            node._next = comm_nodes[0]


def remove_unimportant_nodes_and_reconnect(graph: Graph, kv_caching: bool = False):
    node: Node
    for node in graph.nodes:
        if node.op == "output":
            continue
        if kv_caching and "mha" in node.name:
            node.kwargs["op_info"]["kv_caching"] = True
        if not is_important(node):
            for u in node._input_nodes:
                for v in node.users:
                    if _valid_connection(u, v):
                        connect(u, v)
            for u in node._input_nodes:
                u.users.pop(node)
            for v in node.users:
                v._input_nodes.pop(node)

            remove_node(graph, node)


def seq_id_from_op(op: Node):
    """
    Extract the sequence id from the op.

    Args:
        op (Node): tier linear operation.

    Returns:
        (int): Sequence id
    """
    assert (
        "seq_" in op.name
    ), "you need to run assign_sequence_ids_to_nodes which will be run when you use encode"
    regex_sequence_id = re.findall(r"seq_(\d+)", op.name)
    assert (
        len(regex_sequence_id) == 1
    ), f"trouble identifying sequence id in op {op.name}"
    sequence_id = int(regex_sequence_id[0])
    return sequence_id


def get_op_info(op: Node):
    """
    Returns the additional info stored in the op's kwargs.

    Args:
        op (Node): The operation must be called with the additional
        info dict as a kwarg with the name op_info

    Returns:
        (dict): operation info
    """
    assert len(op.kwargs) > 0, "operation info dict must be given as kwargs"
    op_info: dict = op.kwargs["op_info"]
    return op_info


def _valid_connection(node1: Node, node2: Node):
    if not (is_feature_node(node1) and is_feature_node(node2)):
        return True
    op_info1 = get_op_info(node1)
    op_info2 = get_op_info(node2)
    # if decoder step id exists, it has to be the same
    # its also a valid connection if the decoder sequence ids are different

    """
    1. Create test that generates the graph knowing the groundtruth of
    connections.
        - lm_heads feed into this and that
        - ...
    2. Do this for decoder and encoder-decoder
    3. Ensure that only these connections are allowed by tweaking the
    conditions here.
    """

    if "decoding_id" in op_info1:
        assert "decoding_id" in op_info2, "decoding_id must be also in op_info2"
        edge = (op_info1["decoding_id"], op_info2["decoding_id"])

        # multinomial node can at most connect to nodes in the next decoding step
        if "multinomial" in node1.name and abs(edge[0] - edge[1]) > 1:
            return False

        is_cross = "memory" in edge and edge != ("memory", "memory")
        return (
            is_cross
            or op_info1["token_id"] == op_info2["token_id"]
            or op_info1["decoding_id"] != op_info2["decoding_id"]
        )
    else:
        assert not "decoding_id" in op_info2, "decoding_id cannot be in op_info2"
        return op_info1["token_id"] == op_info2["token_id"]

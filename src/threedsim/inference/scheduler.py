import copy
from math import ceil
from functools import lru_cache
from tqdm import tqdm
import re

from torch.fx.graph import Graph
from torch.fx.node import Node

from ..accelerator import Accelerator, AcceleratorConfig, kv_cache_latency
from ..graph.processing import (
    add_comm_nodes,
    get_op_info,
    prune_and_merge_ops,
    remove_unimportant_nodes_and_reconnect,
    seq_id_from_op,
)
from ..graph.utils import _is_digital, get_source_nodes, is_feature_node, remove_nodes
from ..plotting import plot_event_pipeline, plot_memory_trace
from ..utils import AddressGenerator, TrackEvent


def is_dram_free(mha_op: Node, dram_bandwidth: float):
    op_info = get_op_info(mha_op)
    time_of_loading_from_dram = kv_cache_latency(
        op_info["size"], op_info["seq_len"], dram_bandwidth
    )
    if hasattr(mha_op, "processing_time"):
        # mha_op.processing_time is the remaining time of the operation
        # mha_op.latency is the total time of the operation
        # mha_op.latency - mha_op.processing_time is the time that the operation has been running
        if (mha_op.latency - mha_op.processing_time) < time_of_loading_from_dram:
            return False
    return True


def schedule_execution(
    orig_graph: Graph,
    accelerator: Accelerator,
    copy_and_cleanup_graph: bool = True,
    plot: bool = False,
    plot_dir: str = None,
    communication: bool = True,
    save_memory: bool = True,
    save_peak_memory: bool = True,
):
    accelerator_config = accelerator.config
    events = []

    if copy_and_cleanup_graph:
        # Fix the graph
        print("Removing unimportant nodes...")
        graph = copy.deepcopy(orig_graph)
        prune_and_merge_ops(graph)
        remove_unimportant_nodes_and_reconnect(graph)
        print("...done.")
    else:
        graph = orig_graph

    if communication:
        # add communication nodes
        print("Adding communication nodes...")
        add_comm_nodes(graph)
        print("...done.")

    # Initialize global clock
    current_time = 0

    # Initialize memory and address generator
    memory = {}
    addr_generator = AddressGenerator()
    mem_trace, time_trace = [], []
    peak_mem = -1.0

    # Initialize the energy and latency breakdown dictionaries
    op_keys = [
        "tier_linear",
        "communication",
        "mha",
        "layer_norm",
        "digital_relu",
        "digital_gelu",
        "digital_add",
        "multinomial",
    ]
    break_down_energy = {**{"passive": 0}, **{k: 0 for k in op_keys}}
    break_down_latency = {k: 0 for k in op_keys}
    break_down_energy["mha"] = {"dram": 0, "comp": 0}
    break_down_latency["mha"] = {"dram": 0, "comp": 0}

    # Initialize active energy running sum
    active_energy = 0

    # FLOPS (counted as number of multiply operations for key operations)
    flops = 0

    # Find the initial source nodes of the graph
    print("Picking source nodes...")
    # We assign the input degree of the nodes
    source_nodes = get_source_nodes(graph, assign_edge_degree=True)
    print("...done.")

    # In this loop we schedule the graph and calculate
    # the time, energy and scratchpad memory it needs
    # to execute. This is done using only one graph pass
    start_len = graph._len
    pbar = tqdm(total=start_len)
    while source_nodes != []:
        # Select from the source nodes based on the scheduling heuristic.
        # It also returns which tiles are active for the current step.
        picked_source_nodes, active_tiles = _pick_source_node(
            source_nodes, accelerator=accelerator
        )

        # Assign duration to the picked nodes. If the node is in process
        # (already started but not finished) we don't change its state
        # Then, assign memory dependencies to the children nodes of the picked nodes
        _assign_memory_dependencies_and_latency(
            picked_source_nodes,
            accelerator_config,
            memory,
            addr_generator,
            communication,
            save_memory or save_peak_memory,
        )

        if save_memory or save_peak_memory:
            # Consume the data of the picked nodes to start execution
            _consume_data(picked_source_nodes, memory)

        # Sort operations based on duration. We will move
        # the clock based on the shortest operation (not all of them will finish)
        sorted_ops = sorted(picked_source_nodes, key=lambda x: x.processing_time)
        time_step = sorted_ops[0].processing_time
        current_time += time_step

        # Complete (and pop) the operations that finish within the
        # selected time_step. For the rest, reduce their remaining
        # duration by the time_step amount (midway through execution)
        nodes_to_pop = []
        for n in sorted_ops:
            if n.processing_time == time_step:
                # Node finished so we will pop it
                nodes_to_pop.append(n)

                if save_memory or save_peak_memory:
                    # Put the produced data in the memory
                    # according to the memory dependencies
                    _stage_data(n, memory)

                # Add the active energy of the node to the running sum
                active_energy_of_finished_node = _calculate_energy(
                    n, accelerator_config
                )
                for op_name in op_keys:
                    if op_name in n.name:
                        if op_name == "mha":
                            # need to distinguish between dram and mha time/energy
                            dram_energy, comp_energy = active_energy_of_finished_node
                            active_energy_of_finished_node = dram_energy + comp_energy
                            break_down_energy[op_name]["dram"] += dram_energy
                            break_down_energy[op_name]["comp"] += comp_energy
                            break_down_latency[op_name]["dram"] += n.dram_latency
                            break_down_latency[op_name]["comp"] += n.comp_latency
                        else:
                            break_down_energy[op_name] += active_energy_of_finished_node
                            break_down_latency[op_name] += n.latency
                        break
                active_energy += active_energy_of_finished_node
                flops += _calculate_flops(n, accelerator_config)

                if plot:  # Add an event to plot later
                    _add_event(events, n, current_time)
            else:
                # Node did not finish yet. Reduce the
                # duration of the operation by the time_step amount
                n.processing_time -= time_step

        if save_memory or save_peak_memory:
            memory_utilization = _calculate_memory_utilization(memory)
            if save_peak_memory and memory_utilization > peak_mem:
                peak_mem = memory_utilization

        if save_memory:
            # Calculate the current memory utilization at this time_step
            mem_trace.append(memory_utilization)
            time_trace.append(current_time)

        # Remove the popped nodes from the source node list
        # Add any newly produced source nodes to the list
        for n in nodes_to_pop:
            source_nodes.remove(n)
            for child in n.users:
                if child.op == "output":
                    continue
                child.input_edge_degree -= 1
                if child.input_edge_degree == 0:
                    source_nodes.append(child)

        # Profile it to see if it's slow
        # Remove the nodes that completed their execution
        remove_nodes(graph, nodes_to_pop)

        # update progress bar
        pbar.update(len(nodes_to_pop))

        # Update the running clock of the tiles so that we know
        # how much time have we spent using them compared to the
        # overall execution time
        for tile in active_tiles:
            accelerator.tiles[tile].active_time += time_step

    if plot:
        # Plot the events and the memory trace
        assert plot_dir is not None
        plot_event_pipeline(events=events, plot_dir=plot_dir)
        if save_memory:
            plot_memory_trace(mem_trace, time_trace, plot_dir=plot_dir)

    # Calculate the total energy, passive + active
    passive_energy = current_time * accelerator_config.passive_power
    total_energy = active_energy + passive_energy
    break_down_energy["passive"] = passive_energy

    # return the required time, the scratchpad memory over time
    # the total energy spent, and the number of flops used
    return (
        current_time,
        mem_trace,
        peak_mem,
        total_energy,
        flops,
        break_down_energy,
        break_down_latency,
    )


# worst case O(n), super naive implementation
def _pick_source_node(
    source_nodes: list[Node],
    accelerator: Accelerator,
):
    accelerator_config = accelerator.config
    picked_nodes = []
    tile2op = {}
    mha_ops = {"active": [], "candidate": []}
    dpu_ops = {"active": [], "candidate": []}

    active_tiles = []
    active_mha_units = 0
    active_digital_units = 0
    if accelerator_config.lock_mha_unit_to_layer:
        dict_free_mha_units = {i: None for i in range(accelerator_config.num_mha_units)}

    if accelerator_config.lock_dpu_unit_to_layer:
        dict_free_dpu_units = {
            i: None for i in range(accelerator_config.num_digital_units)
        }

    def get_mha_target(op):
        layer_id = 0 if op.name.split("_")[2] == "seq" else int(op.name.split("_")[2])
        mha_target = layer_id % accelerator_config.num_mha_units
        return mha_target

    def get_dpu_target(op):
        layer_name = [*op.meta["nn_module_stack"]][
            -1
        ]  # This is broken for the multinomial node, TODO fix if we want to use `lock_mha_unit_to_layer` option
        match = re.search(r"layers?\.(\d+)", layer_name)
        if match is None:
            layer_id = 0
        else:
            layer_id = int(match.group(1))
        dpu_target = layer_id % accelerator_config.num_digital_units
        return dpu_target

    # min_seq_id = min([seq_id_from_op(op) for op in source_nodes])

    for op in source_nodes:
        # this is to limit the number of sequences that can be in the
        # pipeline at the same time. caps the max memory
        # seq_id = seq_id_from_op(op)
        # if seq_id >= min_seq_id + 8:
        #     continue

        if "tier_linear" in op.name:
            seq_id = seq_id_from_op(op)
            op_info = get_op_info(op)
            tile_idx = op_info["tile_idx"]
            token_id = op_info["token_id"]

            # That means the tile already started an MVM
            if hasattr(op, "processing_time"):
                if accelerator_config.model_ott:
                    assert (
                        tile_idx not in active_tiles
                    ), "Scheduling error, two MVMs happening on the same tile"
                    tile2op[tile_idx] = [(seq_id, token_id, op)]
                    active_tiles.append(tile_idx)
                else:
                    if tile_idx in tile2op:
                        tile2op[tile_idx].append((seq_id, token_id, op))
                    else:
                        tile2op[tile_idx] = [(seq_id, token_id, op)]
                    if tile_idx not in active_tiles:
                        active_tiles.append(tile_idx)

            if tile_idx in active_tiles:
                # The tile I want is used, so increase the number of conflicts.
                if not "embedding" in op.kwargs["op_info"]["layer_name"]:
                    # op.name wants to access tile number tile_idx, but it's already in use by accelerator.tiles[tile_idx].current_op.name
                    # is accelerator.tiles[tile_idx].current_op.name already in op.conflicted_with? If not, add +1 to the number of conflicts of
                    # accelerator.tiles[tile_idx].num_conflicts and add accelerator.tiles[tile_idx].current_op.name to op.conflicted_with
                    tile_current_op = accelerator.tiles[tile_idx].current_op
                    if not hasattr(op, "conflicted_with"):
                        op.conflicted_with = []
                    if (
                        tile_current_op is not None
                        and not tile_current_op.name in op.conflicted_with
                    ):
                        op.conflicted_with.append(
                            accelerator.tiles[tile_idx].current_op.name
                        )
                        accelerator.tiles[tile_idx].num_conflicts += 1
                continue
            if not tile_idx in tile2op:
                tile2op[tile_idx] = []
            tile2op[tile_idx].append((seq_id, token_id, op))
        elif "communication" in op.name:
            picked_nodes.append(op)
        elif "mha" in op.name:
            # picked_nodes.append(op)
            if hasattr(op, "processing_time"):
                mha_ops["active"].append(op)
                active_mha_units += 1
                if accelerator_config.lock_mha_unit_to_layer:
                    # which MHA unit is this op using?
                    mha_target = get_mha_target(op)
                    # this one is currently being used
                    dict_free_mha_units.pop(mha_target, None)
            else:
                seq_id = seq_id_from_op(op)
                mha_ops["candidate"].append((seq_id, op))
        else:
            if hasattr(op, "processing_time"):
                dpu_ops["active"].append(op)
                active_digital_units += 1
                if accelerator_config.lock_dpu_unit_to_layer:
                    # which DPU unit is this op using?
                    dpu_target = get_dpu_target(op)
                    # this one is currently being used
                    dict_free_dpu_units.pop(dpu_target, None)
            else:
                op_info = get_op_info(op)
                seq_id = seq_id_from_op(op)
                token_id = op_info["token_id"]
                dpu_ops["candidate"].append((seq_id, token_id, op))

    # for each tile idx, sort the entries by seq_id
    # and pick the first one (depth-first)
    for tile_idx in tile2op:
        if accelerator_config.model_ott:
            # sort by the first two elements, namely (seq_id, token_id), seq_id takes presedence
            next_op = sorted(tile2op[tile_idx], key=lambda x: x[:-1])[0][-1]
            picked_nodes.append(next_op)
            accelerator.tiles[tile_idx].current_op = next_op
        else:
            picked_nodes.extend([op_tuple[2] for op_tuple in tile2op[tile_idx]])

    # Add the in-progress dpu and mha operations to the picked nodes
    assert (
        active_digital_units <= accelerator_config.num_digital_units
    ), "More DPU ops than units"
    assert (
        active_mha_units <= accelerator_config.num_mha_units
    ), "More MHA ops than units"
    picked_nodes += dpu_ops["active"]
    picked_nodes += mha_ops["active"]

    # Select from the candidate dpu and mha ops if free units exist
    free_dpu_units = accelerator_config.num_digital_units - active_digital_units
    free_mha_units = accelerator_config.num_mha_units - active_mha_units
    dram_free = True
    for active_mha_op in mha_ops["active"]:
        # check if one op is in the middle of loading data from DRAM
        dram_free = dram_free and is_dram_free(
            active_mha_op, accelerator_config.dram_bandwidth
        )

    if free_dpu_units > 0 and len(dpu_ops["candidate"]) > 0:
        sorted_dpu_ops = sorted(dpu_ops["candidate"], key=lambda x: x[:-1])
        for i in range(min(free_dpu_units, len(sorted_dpu_ops))):
            dpu_op = sorted_dpu_ops[i][-1]
            if accelerator_config.lock_dpu_unit_to_layer:
                dpu_target = get_dpu_target(dpu_op)
                if dpu_target in dict_free_dpu_units:
                    picked_nodes.append(dpu_op)
            else:
                picked_nodes.append(dpu_op)

    if free_mha_units > 0 and len(mha_ops["candidate"]) > 0:
        sorted_mha_ops = sorted(mha_ops["candidate"], key=lambda x: x[:-1])
        for i in range(min(free_mha_units, len(sorted_mha_ops))):
            mha_op = sorted_mha_ops[i][-1]
            if accelerator_config.lock_mha_unit_to_layer:
                mha_target = get_mha_target(mha_op)
                if mha_target in dict_free_mha_units:
                    picked_nodes.append(mha_op)
            else:
                picked_nodes.append(mha_op)

    return picked_nodes, active_tiles


def _assign_memory_dependencies_and_latency(
    nodes: list[Node],
    accel_config: AcceleratorConfig,
    memory: dict[int, dict],
    addr_generator: AddressGenerator,
    communication: bool,
    save_memory: bool,
):
    for node in nodes:
        _assign_latency(node, accel_config)
        if save_memory:
            _assign_memory_dep(
                node, accel_config, memory, addr_generator, communication
            )


def _assign_latency(node: Node, accelerator_config: AcceleratorConfig):
    if hasattr(node, "processing_time"):
        return

    if "tier_linear" in node.name:  # Static time for MVM
        lat = accelerator_config.mvm_latency
    elif "communication" in node.name:  # Always same as the tile's output dimension
        lat = accelerator_config.com_latency(accelerator_config.tier_shape[1])
    elif "multinomial" in node.name:
        lat = accelerator_config.multinomial_latency(
            node.kwargs["op_info"]["vocab_size"]
        )
    else:
        vector_size = _get_size_from_digital_op(node)
        if "mha_" in node.name:  # Function of seq len and size
            kv_caching = False
            if "kv_caching" in node.kwargs["op_info"]:
                kv_caching = node.kwargs["op_info"]["kv_caching"]
            dram_lat, comp_lat = accelerator_config.mha_latency(
                vector_size,
                node.kwargs["op_info"]["seq_len"],
                node.kwargs["op_info"]["causal"],
                node.kwargs["op_info"]["nhead"],
                kv_caching,
            )
            lat = dram_lat + comp_lat
            node.dram_latency = dram_lat
            node.comp_latency = comp_lat
        elif "layer_norm" in node.name:
            lat = accelerator_config.layer_norm_latency(vector_size)
        elif "digital_relu" in node.name:
            lat = accelerator_config.relu_latency(vector_size)
        elif "digital_gelu" in node.name:
            lat = accelerator_config.gelu_latency(vector_size)
        elif "digital_add" in node.name:
            lat = accelerator_config.add_latency(vector_size)
        else:
            raise ValueError("Unknown node type")

    node.latency = lat
    node.processing_time = lat


def _assign_memory_dep(
    node: Node,
    accel_config: AcceleratorConfig,
    memory: dict[int, dict],
    addr_generator: AddressGenerator,
    communication: bool,
):
    if (
        hasattr(node, "output_data_addr")
        or len(node.users) == 0
        or "communication" in node.name
    ):
        return
    # Here we save all the addresses that this node
    # will write its outputs after it's done
    node.output_data_addr = []
    if "mha" in node.name:
        # Considers that ONLY communication nodes exist after the MHA block
        # Also all communication nodes lead to linear layers
        op_info = node.kwargs["op_info"]
        children = (
            node.users
            if not communication
            else [list(n.users.keys())[0] for n in node.users]
        )
        seq_len = op_info["seq_len"]
        if "kv_caching" in op_info and op_info["kv_caching"]:
            seq_len = 1

        num_out_edges = len(children)
        assert num_out_edges % seq_len == 0, "Random edge found?"
        edges_per_token = num_out_edges // seq_len
        d_model = op_info["size"]
        unique_edges_per_token = ceil(d_model / accel_config.tier_shape[0])
        for t in range(seq_len):
            addr = [addr_generator.get_addr() for _ in range(unique_edges_per_token)]
            node.output_data_addr += addr
            sorted_linear_layers_for_token = sorted(
                [n for n in children if n.kwargs["op_info"]["token_id"] == t],
                key=lambda x: x.name.split("_")[2],
            )  # NOTE Can fail if MHA is first op in the graph (never happens)
            col_freq = edges_per_token // unique_edges_per_token
            for i in range(unique_edges_per_token):
                memory[addr[i]] = {"data": d_model, "count": col_freq, "active": False}
                for n in sorted_linear_layers_for_token[
                    i * col_freq : (i + 1) * col_freq
                ]:
                    if not hasattr(n, "input_data_addr"):
                        n.input_data_addr = []
                    n.input_data_addr.append(addr[i])

    elif "tier_linear" in node.name:
        # In a tier op, the output data
        # are the same for all child nodes
        # We only generate one "address"
        # NOTE that we bypass the communication
        # node that will be in the graph
        children = node.users if not communication else list(node.users.keys())[0].users

        # DO NOT add dependencies on the lm head layers (the ones that ONLY lead to multinomial)
        if any(["multinomial" in child.name for child in children]):
            assert all(["multinomial" in child.name for child in children])
            return

        addr = addr_generator.get_addr()
        n_cols = node.kwargs["op_info"]["n_cols"]
        memory[addr] = {"data": n_cols, "count": len(children), "active": False}
        node.output_data_addr = [addr]
        for child in children:
            if not hasattr(child, "input_data_addr"):
                child.input_data_addr = []
            child.input_data_addr.append(addr)
    else:
        vector_size = _get_size_from_digital_op(node)
        children = [
            n if not "communication" in n.name else list(n.users.keys())[0]
            for n in node.users
        ]
        if all([_is_digital(n) for n in children]):
            addr = addr_generator.get_addr()
            memory[addr] = {
                "data": vector_size,
                "count": len(children),
                "active": False,
            }
            node.output_data_addr = [addr]
            for child in children:
                if not hasattr(child, "input_data_addr"):
                    child.input_data_addr = []
                child.input_data_addr.append(addr)
        else:  # All analog or mixed analog/digital
            analog_children = [n for n in children if not _is_digital(n)]
            digital_children = [n for n in children if _is_digital(n)]
            # We first assign memories to the analog children
            unique_analog_edges = ceil(vector_size / accel_config.tier_shape[0])
            addr = [addr_generator.get_addr() for _ in range(unique_analog_edges)]
            node.output_data_addr += addr
            analog_children = sorted(
                analog_children,
                key=lambda x: x.name.split("_")[2],
            )  # NOTE Can fail if MHA is first op in the graph (never happens)
            col_freq = len(analog_children) // unique_analog_edges
            d_model = node.kwargs["op_info"]["size"]
            assert len(analog_children) % unique_analog_edges == 0, "Random edge found"
            for i in range(unique_analog_edges):
                memory[addr[i]] = {
                    "data": d_model,
                    "count": col_freq + len(digital_children),
                    "active": False,
                }
                for n in analog_children[i * col_freq : (i + 1) * col_freq]:
                    if not hasattr(n, "input_data_addr"):
                        n.input_data_addr = []
                    n.input_data_addr.append(addr[i])

            # Then we put the same dependencies on the digital parts
            for n in digital_children:
                if not hasattr(n, "input_data_addr"):
                    n.input_data_addr = []
                n.input_data_addr += addr


def _stage_data(node: Node, memory: dict):
    if not hasattr(node, "output_data_addr"):
        return
    is_feature = is_feature_node(node)
    if is_feature and len(node.output_data_addr) > 1:
        memory[node.output_data_addr[-1]]["active"] = True
    else:
        for addr in node.output_data_addr:
            memory[addr]["active"] = True


def _consume_data(picked_source_nodes: list[Node], memory: dict):
    for node in picked_source_nodes:
        if not hasattr(node, "input_data_addr"):
            continue
        for addr in node.input_data_addr:
            memory[addr]["count"] -= 1  # Reduce the use of the address by one
            # If we finished with the data, pop them from the dict
            if memory[addr]["count"] == 0:
                memory.pop(addr)

        # Remove the input_data since we consumed them
        delattr(node, "input_data_addr")


def _calculate_memory_utilization(memory: dict):
    return sum(
        [vector["data"] for vector in memory.values() if vector["active"] == True]
    )


def _calculate_energy(node: Node, accelerator_config: AcceleratorConfig):
    if "tier_linear" in node.name:
        return accelerator_config.mvm_energy
    elif "communication" in node.name:
        return accelerator_config.com_energy(accelerator_config.tier_shape[1])
    else:
        vector_size = _get_size_from_digital_op(node)
        if "mha" in node.name:  # Function of seq len and size
            kv_caching = False
            if "kv_caching" in node.kwargs["op_info"]:
                kv_caching = node.kwargs["op_info"]["kv_caching"]
            return accelerator_config.mha_energy(
                vector_size,
                node.kwargs["op_info"]["seq_len"],
                node.kwargs["op_info"]["causal"],
                kv_caching,
            )
        elif "layer_norm" in node.name:
            return accelerator_config.layer_norm_energy(vector_size)
        elif "digital_relu" in node.name:
            return accelerator_config.relu_energy(vector_size)
        elif "digital_gelu" in node.name:
            return accelerator_config.gelu_energy(vector_size)
        elif "digital_add" in node.name:
            return accelerator_config.add_energy(vector_size)
        elif "multinomial" in node.name:
            return accelerator_config.multinomial_energy(vector_size)
        else:
            raise ValueError("Unknown node type")


def _calculate_flops(node: Node, accelerator_config: AcceleratorConfig):
    if "tier_linear" in node.name:
        utilization = node.kwargs["op_info"]["utilization"]
        n_rows, n_cols = accelerator_config.tier_shape
        return utilization * (
            n_rows * n_cols + n_rows * (n_cols - 1)
        )  # w/o add: n_rows * n_cols
    elif "communication" in node.name:
        return 0
    else:
        vector_size = _get_size_from_digital_op(node)
        if "mha" in node.name:  # Function of seq len and size
            kv_caching = False
            if "kv_caching" in node.kwargs["op_info"]:
                kv_caching = node.kwargs["op_info"]["kv_caching"]
            return _flops_mha(
                vector_size,
                node.kwargs["op_info"]["seq_len"],
                node.kwargs["op_info"]["causal"],
                kv_caching,
            )
        elif "layer_norm" in node.name:
            return _flops_layer_norm(vector_size)
        elif "digital_relu" in node.name:
            return _flops_relu(vector_size)
        elif "digital_gelu" in node.name:
            return _flops_gelu(vector_size)
        elif "digital_add" in node.name:
            return _flops_add(vector_size)
        elif "multinomial" in node.name:
            return _flops_multinomial(vector_size)
        else:
            raise ValueError("Unknown node type")


@lru_cache(maxsize=4)
def _flops_layer_norm(vector_size: int):
    # flops = vector_size # mean
    # # var (sub the mean, square it, sum them, divide by remaining shape)
    # flops += 3 * vector_size
    # # add eps and running_var, sqrt it
    # flops += 2 * vector_size
    # # For each element, sub running_mean, div by denom
    # flops += 2 * vector_size
    # # For each element, mul by gamma, add beta
    # flops += 2 * vector_size
    # return flops
    return 10 * vector_size


@lru_cache(maxsize=4)
def _flops_sigmoid(vector_size: int):
    # For each element, mul by -1, exp it, add 1, div
    return vector_size * 4


@lru_cache(maxsize=4)
def _flops_gelu(vector_size: int):
    # element-wise scaling, sigmoid, element-wise mult (https://paperswithcode.com/method/gelu)
    return _flops_sigmoid(vector_size) + 2 * vector_size


@lru_cache(maxsize=4)
def _flops_mha(vector_size: int, seq_length: int, causal: bool, kv_caching: bool):
    # qkT
    # how many dot products are performed?
    if kv_caching:
        # we only have one query vector. We need to compute the dot product with all key vectors.
        num_dot_products = seq_length
    else:
        num_dot_products = (
            (seq_length**2 + seq_length) / 2 if causal else seq_length**2
        )

    flops = (
        2 * num_dot_products * vector_size
    )  # each dot prod has vector_size many mults and adds
    # sqrt(d)
    flops += 1 + num_dot_products
    # softmax
    flops += 2 * num_dot_products + (
        num_dot_products - seq_length
    )  # (2 * num_dot_products w/o add)
    # A*V
    if kv_caching:
        flops += 2 * seq_length * vector_size
    else:
        flops += 2 * seq_length**2 * vector_size
    return flops


@lru_cache(maxsize=4)
def _flops_relu(vector_size: int):
    return vector_size  # 1 comparison one flop


@lru_cache(maxsize=4)
def _flops_add(vector_size: int):
    return vector_size  # 1 add one flop, 0 if add doesn't count


@lru_cache(maxsize=4)
def _flops_multinomial(vector_size: int):
    return vector_size  # hardcoded to be vector_size = 1


@lru_cache(maxsize=4)
def _get_size_from_digital_op(node: Node):
    return node.kwargs["op_info"]["size"]


def _add_event(
    event_list: list[TrackEvent],
    node: Node,
    current_time: int,
):
    # Create an event to plot
    if "mha" in node.name:
        sequence_id = seq_id_from_op(node)
        e = TrackEvent(
            time=current_time - node.latency,
            name=node.name,
            token_id=None,
            sequence_id=sequence_id,
            seq_len=node.kwargs["op_info"]["seq_len"],
            operation_duration=node.latency,
        )
        event_list.append(e)
    elif is_feature_node(node):
        name = _get_name(node)
        sequence_id = seq_id_from_op(node)
        op_info = get_op_info(node)
        e = TrackEvent(
            time=current_time - node.latency,
            name=name,
            token_id=op_info["token_id"],
            sequence_id=sequence_id,
            seq_len=op_info["seq_len"],
            operation_duration=node.latency,
        )
        event_list.append(e)
    elif "communication" in node.name:
        return
    else:
        raise ValueError("Unknown node type")


def _get_name(node: Node):
    if "tier_linear" in node.name:
        return get_op_info(node)["layer_name"]
    elif "layer_norm" in node.name:
        return "norm"
    elif "digital_add" in node.name:
        return "add"
    elif "digital_relu" in node.name:
        return "relu"
    elif "digital_gelu" in node.name:
        return "gelu"
    elif "multinomial" in node.name:
        return "multinomial"

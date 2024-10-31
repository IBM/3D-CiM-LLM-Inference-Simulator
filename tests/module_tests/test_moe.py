import torch

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.graph import prune_and_merge_ops, remove_unimportant_nodes_and_reconnect
from threedsim.inference import schedule_execution, trace_model
from threedsim.modules import MoELayer
from threedsim.modules.base import assign_acc, fill_name_fields, make_traceable
from threedsim.plotting import plot_graph

device = "cpu"


class MoE(torch.nn.Module):
    def __init__(self, moe_layer):
        super().__init__()
        self.moe_layer = moe_layer

    def forward(self, x):
        return self.moe_layer(x, op_info={"token_id": 0, "seq_len": 1})


if __name__ == "__main__":
    plot_graphs = False
    num_sequences = 1
    config = AcceleratorConfig(tiles=10, tiers=21, tier_shape=(512, 512))
    acc = Accelerator(config, device=device)
    moe_kwargs = {
        "dim_feedforward": 512,
        "d_model": 512,
        "k": 2,
        "num_experts": 3,
        "density": torch.randn(3, 3).abs(),
    }
    moe_layer = MoELayer(**moe_kwargs, device=device)
    assign_acc(moe_layer, acc)

    moe_layer.router.set_mapping([[(0, 1, 1, 512, 512)]])
    moe_layer.experts[0].ffn1.set_mapping([[(0, 2, 1, 512, 512)]])
    moe_layer.experts[0].ffn2.set_mapping([[(0, 3, 1, 512, 512)]])
    moe_layer.experts[1].ffn1.set_mapping([[(0, 4, 1, 512, 512)]])
    moe_layer.experts[1].ffn2.set_mapping([[(0, 5, 1, 512, 512)]])
    moe_layer.experts[2].ffn1.set_mapping([[(0, 6, 1, 512, 512)]])
    moe_layer.experts[2].ffn2.set_mapping([[(0, 7, 1, 512, 512)]])

    fill_name_fields(moe_layer)
    make_traceable(moe_layer, is_traceable=True)

    model = MoE(moe_layer)

    symbolic_traced = trace_model(model, num_sequences)
    execution_time, memory, peak_memory, energy, flops, _, _ = schedule_execution(
        symbolic_traced.graph,
        accelerator=model.moe_layer.accelerator,
        plot=True,
        plot_dir="results",
    )
    print(f"Execution took {execution_time} units")
    print(f"Required {peak_memory} bytes of scratchpad memory")
    print(f"Spent {energy} xJ of energy")

    if plot_graphs:
        plot_graph(symbolic_traced, "results/unpruned.svg")
        prune_and_merge_ops(graph=symbolic_traced.graph)
        plot_graph(symbolic_traced, "results/pruned.svg")
        remove_unimportant_nodes_and_reconnect(graph=symbolic_traced.graph)
        plot_graph(symbolic_traced, "results/final.svg")

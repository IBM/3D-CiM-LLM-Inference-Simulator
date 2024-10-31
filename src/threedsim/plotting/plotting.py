import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from torch.fx import passes
from torch.fx.graph import Graph

from ..utils import TrackEvent
from .params import load_originlike_rcparams

load_originlike_rcparams()

layer2color = {
    "q_proj": "y",  # yellow
    "k_proj": "r",  # red
    "v_proj": "b",  # blue
    "out_proj": "c",  # cyan
    "ffn1": "g",  # green
    "ffn2": "m",  # magenta
    "mha": "k",  # black
    "embedding": "#008080",  # teal
    "lm_head": "brown",  # brown
    "norm": "pink",  # pink
    "add": "navy",  # navy
    "relu": "orange",  # orange
    "multinomial": "grey",  # grey
}


def assign_color(event: TrackEvent):
    for name in layer2color:
        if name in event.name:
            return layer2color[name]
    # return black if not found
    return "k"


def plot_graph(graph: Graph, file_path: str, label: str = ""):
    _create_dir(os.path.dirname(file_path))
    assert len(file_path) > 4 and file_path[-4:] == ".svg", "Format must be svg"
    g = passes.graph_drawer.FxGraphDrawer(graph, label)
    with open(file_path, "wb") as f:
        f.write(g.get_dot_graph().create_svg())


def plot_event_pipeline(events: list[TrackEvent], plot_dir: str):
    _create_dir(plot_dir)
    save_path = os.path.join(plot_dir, "execution_flow.pdf")
    pp = PdfPages(save_path)
    max_sequence_length = max([e.seq_len for e in events])
    max_time = max([e.time + e.operation_duration for e in events])
    max_sequence_id = max([e.sequence_id for e in events])
    # x: [0, max_time]
    # y: [0, max_token_id] for all plots (i.e. for all sequences)
    # one token plots from X: [start, stop] Y: [token_id, token_id+1]

    f = lambda m, c: plt.plot(
        [],
        [],
        marker=m,
        markersize=16,
        markerfacecolor=c,
        color=c,
        alpha=0.3,
        ls="none",
    )[0]
    labels = [k for k in layer2color]

    for sequence_id in range(max_sequence_id + 1):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_title(f"Sequence {sequence_id}")
        ax.set_ylabel("Token ID")
        ax.set_xlabel("Time (discrete steps)")
        handles = [f("s", v) for v in layer2color.values()]
        events_seq_id = [e for e in events if e.sequence_id == sequence_id]
        boxes = []
        for e in events_seq_id:
            start = e.time

            if e.token_id is None:  # mha
                boxes.append(
                    Rectangle(
                        xy=(start, 0),
                        width=e.operation_duration,
                        height=max_sequence_length,
                        color=assign_color(e),
                        alpha=0.5,
                    )
                )
            else:  # linear
                boxes.append(
                    Rectangle(
                        xy=(start, e.token_id),
                        width=e.operation_duration,
                        height=1.0,
                        color=assign_color(e),
                        alpha=0.5,
                    )
                )

        ax.add_collection(PatchCollection(boxes, match_original=True))
        ax.grid(axis="y")
        ax.set_xlim([0, max_time])
        ax.set_ylim([0, max_sequence_length])
        ax.legend(
            handles,
            labels,
            framealpha=1,
            loc=2,
            frameon=False,
            bbox_to_anchor=(1.0, 1.0),
        )
        ax.tick_params(left=True, right=True, bottom=True, top=True, which="major")
        ax.tick_params(bottom=True, top=True, which="minor")
        ax.tick_params(left=False, right=False, which="minor")
        fig.tight_layout()

    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        if fig.get_axes():
            fig.savefig(pp, format="pdf")
        plt.close(fig)
    pp.close()


def plot_memory_trace(memory_trace: list[int], time_trace: list[int], plot_dir: str):
    _create_dir(plot_dir)
    save_path = os.path.join(plot_dir, "memory_usage.pdf")
    fig, ax = plt.subplots(figsize=(6.4 * 5, 4.8))
    ax.plot(time_trace, memory_trace, "-o", color="k", markersize=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Memory Usage (Bytes)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _create_dir(dir):
    if dir == "":
        return
    if not os.path.exists(dir):
        os.makedirs(dir)

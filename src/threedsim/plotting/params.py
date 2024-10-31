from matplotlib import RcParams, rcParams
from matplotlib.pyplot import cycler


def load_originlike_rcparams(rc_params: RcParams | None = None):
    if rc_params is None:
        rc_params = rcParams
    plot_color_cycle = cycler(
        "color",
        [
            "#9b59b6",
            "#3498db",
            "#95a5a6",
            "#e74c3c",
            "#34495e",
            "#2ecc71",
            "#1E2460",
            "#B5B8B1",
            "#734222",
            "#A52019",
        ],
    )
    rc_params.update(
        {
            "font.size": 16,
            "axes.linewidth": 1.1,
            "axes.labelpad": 10,
            "axes.xmargin": 0.0,
            "axes.ymargin": 0.0,
            "axes.prop_cycle": plot_color_cycle,
            "figure.figsize": (6.4, 4.8),
            "figure.subplot.left": 0.177,
            "figure.subplot.right": 0.946,
            "figure.subplot.bottom": 0.156,
            "figure.subplot.top": 0.965,
            "lines.markersize": 10,
            "lines.markerfacecolor": "none",
            "lines.markeredgewidth": 0.8,
            "axes.autolimit_mode": "round_numbers",
            "xtick.major.size": 7,
            "xtick.minor.size": 3.5,
            "xtick.major.width": 1.1,
            "xtick.minor.width": 1.1,
            "xtick.major.pad": 5,
            "xtick.minor.visible": True,
            "xtick.top": True,
            "xtick.direction": "in",
            "ytick.major.size": 7,
            "ytick.minor.size": 3.5,
            "ytick.major.width": 1.1,
            "ytick.minor.width": 1.1,
            "ytick.major.pad": 5,
            "ytick.minor.visible": True,
            "ytick.right": True,
            "ytick.direction": "in",
        }
    )

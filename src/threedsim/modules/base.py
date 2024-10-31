import torch

from ..accelerator import Accelerator, Tier


class BaseModule:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator: Accelerator = None
        self.name: str = ""
        self.traceable: bool = False

    def set_name(self, name):
        self.name = name


def set_weight(layer: BaseModule, weight: torch.Tensor):
    """
    Given a layer (linear or embedding), chunks the
    weight up into the tier shape and assigns it
    to the correct (mapped) tiers for this layer.

    Args:
        layer (BaseModule): Layer of which we want to assign the weights.
        weight (torch.Tensor): Weight matrix that we want to assign.
    """
    assert hasattr(layer, "mapping"), "is the layer a linear or embedding layer?"
    assert hasattr(layer, "weight"), "layer must be Linear"
    assert layer.mapping is not None, "must be mapped"
    # assign the weight
    assert layer.weight.shape == weight.shape, "shapes don't match"
    layer.weight.data = weight
    n_rows, n_cols = layer.accelerator.config.tier_shape
    # chunk it vertically
    vertically_chunked = weight.split(n_rows)
    assert len(vertically_chunked) == len(
        layer.mapping
    ), "chunked vertical does not match vertical number in mapping"
    for row_idx, row_block in enumerate(vertically_chunked):
        # chunk the row block horizontally
        hor_blocks = row_block.split(n_cols, dim=-1)
        assert len(hor_blocks) == len(layer.mapping[0])
        for col_idx, hor_block in enumerate(hor_blocks):
            nr, nc = hor_block.shape
            tile_idx, tier_idx, utilization, n_rows, n_cols = layer.mapping[row_idx][
                col_idx
            ]
            tier: Tier = layer.accelerator.tiles[tile_idx].tiers[tier_idx]
            tier.weight[:nr, :nc] = hor_block


def assign_acc(model: torch.nn.Module, accelerator: Accelerator) -> None:
    """
    Goes through modules in model and assigns env and accelerator to ones
    that are instance of BaseModule.

    Args:
        model (torch.nn.Module): Model to assign to.
        accelerator (Acclelerator): Accelerator to be used.
    """
    for module in model.modules():
        if isinstance(module, BaseModule):
            module.accelerator = accelerator


def make_traceable(model: torch.nn.Module, is_traceable: bool):
    for module in model.modules():
        if hasattr(module, "traceable"):
            module.traceable = is_traceable


def make_use_linear(model: torch.nn.Module, use_linear: bool):
    for module in model.modules():
        if hasattr(module, "use_linear"):
            module.use_linear = use_linear


def fill_name_fields(model: torch.nn.Module):
    """
    Goes through the modules of the model and
    assigns the name using the set_name method.
    Important: All models must inherit from
    BaseModule. Otherwise, set_name is not implemented.
    Important: Mapping has to be assigned prior
    to calling this method.

    Args:
        model (torch.nn.Module): Model to be named.
    """
    for name, module in model.named_modules():
        if hasattr(module, "set_name"):
            module.set_name(name)

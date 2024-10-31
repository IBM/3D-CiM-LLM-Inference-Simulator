import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union


def flatten(xs):
    """
    Flatten arbitrary nested list, but stop at str, bytes, tuple

    Args:
        xs (list): List to be flattened.

    Yields:
        list: 1-D list.
    """
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes, tuple)):
            yield from flatten(x)
        else:
            yield x


class Strategy(Enum):
    """
    The different map strategies supported.
    GREEDY: For each tile, pick the next one that has the least
        amount of used tiers. Operates in-order of the layers and
        always picks the first tile available. I.e. if tiles 1,2,3 have
        the least tiers used, will use tile 1.
    GREEDY_RANDOM: Same as greedy, but traverses the layers randomly. Also
        picks the next tile at random. In the example above we would
        pick on of 1,2,3 at random.
    GREEDY_TOP2: Exploits probability distribution of top-2 routing.
    MOE_SEPARATED: Maps experts of MoE layers to same set of tiles.
    """

    GREEDY_IN_ORDER = auto()
    GREEDY_RANDOM = auto()
    GREEDY_UTIL = auto()
    GREEDY_TOP2 = auto()
    MOE_SEPARATED = auto()


@dataclass
class MapStrategy:
    """
    MapStrategy that can be specified in the run method or in the Mapper.
    The strategy determines how the layers are mapped. split_ffn determines
    whether the FFN-in and FFN-out are mapped to the same set of tiles
    or not.
    """

    strategy: Strategy
    split_ffn: bool
    stack_embedding: bool


def num_tiles_for_shape(inp_shape: tuple[int], tier_shape: tuple[int]):
    """
    Compute the number of tiles needed for a given shape.
    Args:
        inp_shape (tuple[int]): Shape of matrix.
        tier_shape (tuple[int]): Shape of one tier.
    Returns:
        (int): Number of tiles needed.
    """
    inp_rows, inp_cols = inp_shape
    tier_rows, tier_cols = tier_shape
    num_x = math.ceil(inp_rows / tier_rows)
    num_y = math.ceil(inp_cols / tier_cols)
    return num_x * num_y

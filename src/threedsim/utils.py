import logging
import random
from dataclasses import dataclass

import numpy as np
import torch


def add_dependency(*args):
    """
    IMPORTANT: After import, wrap using fx.wrap(add_dependency)
    Used for making sure that part of a code is executed
    """
    return args


def cross_dependency(*args):
    """
    IMPORTANT: After import, wrap using fx.wrap(add_dependency)
    Used for making sure that part of a code is executed.
    This dependency is replaced by an all-to-all
    connection from the input- to output-dependencies.
    """
    return args


def digital_add(a, b, op_info: dict = None):
    """
    Adds a and b together.

    Returns:
        Sum of a and b.
    """
    return a + b

def multinomial(probs, num_samples, op_info: dict = None):
    """
    Samples from a probability distribution

    Returns:
        Samples from the given probability distribution
    """
    return torch.multinomial(probs, num_samples=num_samples)


def get_logger(name):
    """
    Instantiate a logger with date format.

    Args:
        name (str): Name of the logger.

    Returns:
        (Logger): The logger. Can be used as logger.info(...)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


@dataclass
class TrackEvent:
    """
    Class that stores meta information for events. Used for tracking events.
    """

    time: int
    name: str
    # is_dispatch: bool
    token_id: int
    sequence_id: int
    seq_len: int
    # is_stop: bool
    operation_duration: int


class AddressGenerator:
    def __init__(self) -> None:
        self.next_addr = 0

    def get_addr(self) -> int:
        ret_addr = self.next_addr
        self.next_addr += 1
        return ret_addr


def get_used_tiles(layers: list):
    """
    Get the tiles used by a list of AnalogLinear layers.

    Args:
        layers (List): List of AnalogLinear layers.

    Returns:
        List[int]: Tile indices (ints) used by the layers. Each entry is unique.
    """
    used_tiles = [list(l.tile_tier_dict.keys()) for l in layers]
    used_tiles = np.unique([l for sublist in used_tiles for l in sublist])
    return used_tiles


def set_seed(seed: int):
    """
    Set the seed to a specified value.

    Args:
        seed (int): The seed.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

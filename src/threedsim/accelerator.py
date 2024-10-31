import math
from functools import partial
from typing import Callable
from functools import lru_cache

import torch
from torch import Tensor

from .utils import get_logger


def kv_cache_latency(vector_size: int, seq_length: int, dram_bandwidth: float) -> float:
    """
    Calculate the latency for loading the KV cache in nano seconds.
    We assume dram_bandwidth GB/s bandwidth.
    """
    # one for key, one for value
    # we assume the KV-cache can be quantized to 8-bit
    num_bytes_in_cache = 2 * seq_length * vector_size
    time_to_load = 1e9 * ((num_bytes_in_cache / 1e9) / dram_bandwidth)
    return math.ceil(time_to_load)  # round up to the nearest ns


@lru_cache(maxsize=4)
def calculate_mha_latency(
    vector_size: int,
    seq_length: int,
    causal: bool,
    num_heads: int,
    kv_caching: bool,
    model_sram: bool,
    dram_bandwidth: int,
):
    """
    Calculates the latency for doing multi-head-attention
    in nano seconds.

    Args:
        vector_size (int): Length of the input vectors (d_model).
        seq_length (int): Sequence length (number of tokens).
        causal (bool): If causal attention is used. Essentially halves the number of ops.
        num_heads (int): Number of heads for the attention.
        kv_caching (bool): If we are using KV-caching.
        model_sram (bool): If we are using SRAM in the model.
        dram_bandwidth (int): Bandwidth of the DRAM in GB/s.

    Returns:
        (float, float): KV-cache latency in ns, MHA latency in ns.
    """
    return 1.0, 1.0


@lru_cache(maxsize=4)
def calculate_mha_energy(
    vector_size: int,
    seq_length: int,
    causal: bool,
    kv_caching: bool,
    model_sram: bool,
    dram_bandwidth: float,
    dram_active_power: float,
):
    """
    Calculate the energy consumed by performing multi-head attention.

    Args:
        vector_size (int): Length of the input vectors (d_model).
        seq_length (int): Sequence length (number of tokens).
        causal (bool): If causal attention is used. Essentially halves the number of ops.
        kv_caching (bool): If we are using KV-caching.
        model_sram (bool): If we are using SRAM in the model.
        dram_bandwidth (float): Bandwidth of the DRAM in GB/s.
        dram_active_power (float): Active power of the DRAM in W.

    Returns:
        (float, float): Energy consumed by DRAM in nJ, Energy consumed by the MHA comp in nJ.
    """
    return 1.0, 1.0


class AcceleratorConfig:
    """
    tiles (int): How many 3D tiles the accelerator has.
    tiers (int): How many 3D tiers per tile does the accelerator have.
    tier_shape (tuple[int]): What is the shape of each tier. E.g. (512,256).
    """

    def __init__(
        self,
        tiles: int,
        tiers: int,
        tier_shape: tuple[int, int],
        num_digital_units: int = 4,
        num_mha_units: int = 1,
        mvm_latency: int = 100,
        mvm_energy: float = 10,
        lock_mha_unit_to_layer: bool = False,
        lock_dpu_unit_to_layer: bool = False,
        model_sram: bool = True,
        model_ott: bool = True,
        kv_caching: bool = False,
        dram_bandwidth: float = 5.332,
        dram_active_power: float = 0.2467,
        dram_inactive_power: float = 0.1297,
    ):
        self.tiles = tiles
        self.tiers = tiers
        self.tier_shape = tier_shape
        self.num_digital_units = num_digital_units
        self.num_mha_units = num_mha_units
        self.mvm_latency = mvm_latency
        self.lock_mha_unit_to_layer = lock_mha_unit_to_layer
        self.lock_dpu_unit_to_layer = lock_dpu_unit_to_layer
        self.model_sram = model_sram
        self.model_ott = model_ott
        self.kv_caching = kv_caching
        self.dram_bandwidth = dram_bandwidth  # GB/s
        self.dram_active_power = dram_active_power  # W
        self.dram_inactive_power = dram_inactive_power  # W

        self.mha_latency: Callable[[int, int, bool], int] = partial(
            calculate_mha_latency,
            model_sram=model_sram,
            dram_bandwidth=self.dram_bandwidth,
        )
        # Unit [ns] this is a function of the vector_size
        self.layer_norm_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 50
        )
        # Normally it's inside FMA blocks, we'll put a phantom 1 ns latency (or remove it if it messes scheduling)
        self.relu_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 1
        )
        # Unit [ns] this is a function of the vector size
        self.gelu_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 10
        )
        # Unit [ns] this is a function of the vector size
        self.add_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 10
        )
        # Unit [ns] this is a function of the vector size
        self.com_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 10.0
        )
        # Time to sample from softmax dist. in decoding step
        self.multinomial_latency: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 10
        )

        # Energy numbers (in nJ)
        self.mvm_energy: float = mvm_energy  # 0.1 W power, 100 ns integration
        self.mha_energy: Callable[[int, int, bool], float] = partial(
            calculate_mha_energy,
            model_sram=model_sram,
            dram_bandwidth=self.dram_bandwidth,
            dram_active_power=self.dram_active_power,
        )
        # Unit [nJ] this is a function of the vector_size
        self.layer_norm_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: 10.0
        )

        # Unit [nJ] this is a function of the vector_size
        self.relu_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: 10.0
        )

        # Unit [nJ] this is a function of the vector_size
        self.gelu_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: 10.0
        )

        # Unit [nJ] this is a function of the vector_size
        self.add_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: 10.0
        )

        # Unit [nJ] this is a function of the vector_size (assumes 8 bits to be transferred)
        self.com_energy: Callable[[int], float] = lru_cache(maxsize=4)(
            lambda vector_size: 1e-5 * (vector_size * 8)
        )

        self.multinomial_energy: Callable[[int], int] = lru_cache(maxsize=4)(
            lambda vector_size: 0.0
        )

        # Passive power (in W), adapt to how much power your units use
        self.passive_power: float = (num_mha_units + num_digital_units) * 1.0
        if self.kv_caching:
            self.passive_power += self.dram_inactive_power


class Accelerator:
    def __init__(self, config: AcceleratorConfig, device: str):
        """
        Acclelerator class that holds list of Tiles. Each Tile holds
        list of Tiers.

        Args:
            config (AcceleratorConfig): Configuration for accelerator.
        """
        self.config: AcceleratorConfig = config
        # - Initialize the blocks of the accelerator here
        # tiles (this is a resource), tiers, DPUs (resource)
        self.tiles: list = [
            Tile(
                config.tiers,
                tier_shape=config.tier_shape,
                name=f"tile_{idx}",
                accelerator_config=config,
                device=device,
            )
            for idx in range(config.tiles)
        ]


class Tile:
    def __init__(
        self,
        n_tiers: int,
        tier_shape: tuple[int],
        name: str,
        accelerator_config: AcceleratorConfig,
        device: str,
    ):
        """
        Tile class used in the Accelerator.

        Args:
            n_tiers (int): Number of tiers per Tile.
            tier_shape (tuple[int]): Shape of each tier.
            name (str): Name of the tile.
            accelerator_config (AcceleratorConfig): Configuration of the accelerator.
            device: (str): Device to be used.
        """
        self.logger = get_logger("Tile")
        self.tiers: list = [Tier(tier_shape, device=device) for _ in range(n_tiers)]
        self.name = name
        self.accelerator_config = accelerator_config
        # Number of times I wanted to use the tile, but it was
        # used by another MVM. It is calculated during runtime.
        self.num_conflicts: int = 0
        # How much time have we spent operating the tile, to be
        # used to calculate the mapping efficiency
        self.active_time: int = 0
        self.current_op = None


@torch.fx.wrap
def tier_linear(token, weight, op_info):
    return torch.nn.functional.linear(token, weight.T)


class Tier:
    def __init__(self, tier_shape: tuple[int], device: str):
        """
        Tier class that is used to perform a single MVM.

        Args:
            tier_shape (tuple[int]): Shape of the tier. E.g. (512,512).
            latency (int): Latency for executing one MVM.
            device: (str): Device to be used.
        """
        self.n_rows, self.n_cols = tier_shape
        self.name = ""
        self.mapping = None
        self.used = False
        self.logger = get_logger("Tier")
        self.weight = torch.randn(tier_shape, device=device)
        self.traceable = True
        self.is_mapped = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, token: Tensor, op_info: dict = None):
        assert self.traceable or token.ndim == 1, "token must have only one dim"
        d_token = token.numel()
        token = torch.nn.functional.pad(token, (0, self.n_rows - d_token))
        return tier_linear(token, self.weight, op_info=op_info)

    def set_name(self, name):
        self.name = name

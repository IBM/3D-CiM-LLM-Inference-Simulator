import math
from copy import deepcopy
from random import shuffle
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from ..accelerator import Accelerator
from ..modules import Embedding, Linear, MoELayer
from .utils import MapStrategy, Strategy, flatten, num_tiles_for_shape


class MappingException(Exception):
    pass


class Mapper:
    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        map_strategy: MapStrategy,
        density: Optional[torch.Tensor] = None,
    ):
        """
        Mapper class used for mappig a network.
        Args:
            accelerator (Accelerator): Accelerator.
            model (torch.nn.Module): Model to be mapped.
            map_strategy (MapStrategy): MapStrategy to be used.
            density (torch.Tensor): Density used to top-2 routing with calibrated porbabilities.
        """
        self.accelerator = accelerator
        self.model = model
        self.map_strategy = map_strategy
        self.tile_sum = np.zeros(self.accelerator.config.tiles, dtype=int)
        self.weighted_tile_sum = np.zeros(self.accelerator.config.tiles, dtype=float)
        self.density = density

    def num_params(self):
        """Number of parameter needed by transformer."""
        return sum([p.numel() for p in self.model.parameters() if p.ndim == 2])

    def num_tiles_required(self):
        """Number of tiles that would suffice for mapping the whole network."""
        num_tiers = sum(
            [
                num_tiles_for_shape(p.shape, self.accelerator.config.tier_shape)
                for p in self.model.parameters()
                if p.ndim == 2
            ]
        )
        num_tiles = math.ceil(num_tiers / self.accelerator.config.tiers)
        return num_tiles

    def get_next_k(self, mapped_expert_indices: list[int], S: list[int], E: int):
        """
        Given mapped experts E and the mapped experts S in the
        current planar, find the next expert k that should be
        mapped greedily next.
        Args:
            mapped_expert_indices (list[int]): Experts that are already mapped.
            S (list[int]): Experts mapped in the current planar. S in mapped_expert_indices holds.
            E (int): Number of experts to be mapped in total.
        Returns:
            (int): Next expert to be mapped.
        """
        if E == 1:
            return 0
        max_pi_pS = -1
        max_i = -1
        P = self.density
        for i in range(E):
            if i in mapped_expert_indices or i in S:
                continue
            p_i = P[i].sum()  # - Sum across columns to get P(k1=i)
            if len(S) == 0:
                p_S = 1.0
            else:
                p_S = np.sum([P[i, j] for j in S])
            if p_i * p_S > max_pi_pS:
                max_pi_pS = p_i * p_S
                max_i = i
        return max_i

    def get_low_prior(self, mapped_expert_indices: list[int], S: list[int], E: int):
        """
        Get expert that has low prior probability of being selected.
        This is for the overlap case when an expert spans to planars.
        The expert would block two planars, so we select the one
        that has the lowest prob. of being used.
        Args:
            mapped_expert_indices (list[int]): Experts that are already mapped.
            S (list[int]): Experts mapped in the current planar. S in mapped_expert_indices holds.
            E (int): Number of experts to be mapped in total.
        Returns:
            (int): Expert index with lowest prior prob. that is not mapped yet.
        """
        if E == 1:
            return 0
        min_i = -1
        min_pi = np.inf
        P = self.density
        for i in range(E):
            if i in mapped_expert_indices or i in S:
                continue
            p_i = P[i].sum()  # - Sum across columns to get P(k1=i)
            if p_i < min_pi:
                min_pi = p_i
                min_i = i
        return min_i

    def get_expert_indices_map_order(
        self, offset: int, E: int, d_model: int, dim_feedforward: int
    ):
        """
        If the strategy is GREEDY_TOP2, returns the order in which to
        map the experts so that the overlap is minimized that is given
        by the probability distribution.
        Else, a simple list of the range of number of experts is returned.
        Args:
            offset (int): What is the next tile index that would be used for
                mapping in a greedy fashion? Example: [1,0,0] -> 1, [1,1,0] -> 2.
            E (int): Number of experts to be mapped.
        Returns:
            List[int]: List of expert indices from 0 to E-1.
        """
        if self.map_strategy.strategy == Strategy.GREEDY_TOP2:
            # - Determine the order of the optimal greedy mapping of the experts
            planar = num_tiles_for_shape(
                (d_model, dim_feedforward), self.accelerator.config.tier_shape
            )
            if self.map_strategy.split_ffn:
                # - We split the FFN, so we need two times the planar size.
                planar *= 2

            mapped_expert_indices = []
            while set(mapped_expert_indices) != set([*range(E)]):
                S = []
                num_experts_to_be_mapped = math.ceil(
                    (self.accelerator.config.tiles - offset) / planar
                )
                for expert_index in range(num_experts_to_be_mapped):
                    if (
                        expert_index == num_experts_to_be_mapped - 1
                        and ((self.accelerator.config.tiles - offset) / planar) % 1
                        > 0.0
                    ):
                        # - This expert will get split across two tiles. Choose one with low prior.
                        i = self.get_low_prior(mapped_expert_indices, S, E)
                    else:
                        i = self.get_next_k(mapped_expert_indices, S, E)
                    if i == -1:
                        break
                    assert (
                        not i in mapped_expert_indices
                    ), "Adding index that is already in the list."
                    S.append(i)
                    mapped_expert_indices.append(i)
                    # - Avoid overshooting
                    if set(mapped_expert_indices) == set([*range(E)]):
                        break

                offset = (
                    num_experts_to_be_mapped * planar + offset
                ) % self.accelerator.config.tiles
        else:
            mapped_expert_indices = [*range(E)]
        return mapped_expert_indices

    def map_network(self):
        """
        Maps the network given by transformer config.
        Raises:
            MappingException: If unknown routing strategy is used.
        Returns:
            (dict): Configuration of the mapping.
        """
        self.gen_module_pairs()
        if self.map_strategy.strategy == Strategy.GREEDY_RANDOM:
            # - Shuffle the order in which we map greedily
            shuffle(self.module_pairs)
        elif self.map_strategy.strategy == Strategy.MOE_SEPARATED:
            # put the MoE modules into the front. this makes
            # mapping possible for edge cases
            moe_modules = [m for m in self.module_pairs if isinstance(m, MoELayer)]
            other_modules = [
                m for m in self.module_pairs if not isinstance(m, MoELayer)
            ]
            self.module_pairs = moe_modules + other_modules

        for module in tqdm(self.module_pairs, disable=True):
            if isinstance(module, Linear):
                mapping, _ = self.shape_to_mapping(
                    inp_shape=module.weight.shape,
                    utilization=1.0,
                )
                module.set_mapping(mapping=mapping)

            elif isinstance(module, Embedding):
                test_shape = (
                    self.accelerator.config.tier_shape[0],
                    module.weight.shape[-1],
                )
                utilization = min(
                    1.0, self.accelerator.config.tier_shape[0] / module.weight.shape[0]
                )
                mapping, potential_tiles = self.shape_to_mapping(
                    inp_shape=test_shape,
                    utilization=utilization,
                    dry_run=True,
                )
                # either we stack the embedding on top of each other or
                # we spread it out.
                mapping, _ = self.shape_to_mapping(
                    inp_shape=module.weight.shape,
                    utilization=utilization,
                    tile_indices=potential_tiles
                    if self.map_strategy.stack_embedding
                    else None,
                )
                module.set_mapping(mapping)

            elif isinstance(module, MoELayer):
                # first map the router
                mapping, _ = self.shape_to_mapping(
                    inp_shape=module.router.weight.shape,
                    utilization=1.0,
                )
                module.router.set_mapping(mapping=mapping)

                # then the experts
                ffn1s = [e.ffn1 for e in module.experts]
                ffn2s = [e.ffn2 for e in module.experts]
                if self.map_strategy.strategy == Strategy.MOE_SEPARATED:
                    if self.map_strategy.split_ffn:
                        # first map all ffn_ins
                        ffn1s: List[Linear]
                        mapping, tiles_in = self.shape_to_mapping(
                            inp_shape=ffn1s[0].weight.shape,
                            utilization=1 / module.num_experts,
                            is_repeated=module.num_experts,
                        )
                        ffn1s[0].set_mapping(mapping=mapping)
                        for idx in range(1, module.num_experts):
                            mapping, _ = self.shape_to_mapping(
                                inp_shape=ffn1s[idx].weight.shape,
                                utilization=1 / module.num_experts,
                                tile_indices=tiles_in,
                            )
                            ffn1s[idx].set_mapping(mapping=mapping)

                        # then map all ffn_outs
                        mapping, tiles_out = self.shape_to_mapping(
                            inp_shape=ffn2s[0].weight.shape,
                            utilization=1 / module.num_experts,
                            is_repeated=module.num_experts,
                        )
                        ffn2s[0].set_mapping(mapping=mapping)
                        for idx in range(1, module.num_experts):
                            mapping, _ = self.shape_to_mapping(
                                inp_shape=ffn2s[idx].weight.shape,
                                utilization=1 / module.num_experts,
                                tile_indices=tiles_out,
                            )
                            ffn2s[idx].set_mapping(mapping=mapping)
                    else:
                        # no split
                        mapping, tiles_in = self.shape_to_mapping(
                            inp_shape=ffn1s[0].weight.shape,
                            utilization=1 / module.num_experts,
                            is_repeated=2 * module.num_experts,
                        )
                        ffn1s[0].set_mapping(mapping=mapping)
                        for idx in range(1, module.num_experts):
                            mapping, _ = self.shape_to_mapping(
                                inp_shape=ffn1s[idx].weight.shape,
                                utilization=1 / module.num_experts,
                                tile_indices=tiles_in,
                            )
                            ffn1s[idx].set_mapping(mapping=mapping)
                        for idx in range(module.num_experts):
                            mapping, _ = self.shape_to_mapping(
                                inp_shape=ffn2s[idx].weight.shape,
                                utilization=1 / module.num_experts,
                                tile_indices=tiles_out,
                            )
                            ffn2s[idx].set_mapping(mapping=mapping)

                elif self.map_strategy.strategy in [
                    Strategy.GREEDY_IN_ORDER,
                    Strategy.GREEDY_UTIL,
                    Strategy.GREEDY_RANDOM,
                    Strategy.GREEDY_TOP2,
                ]:
                    # - This is the last index that has one more tier used. It wraps around when we have the same number.
                    offset = (
                        np.where(self.tile_sum == self.tile_sum.max())[0][-1] + 1
                    ) % len(self.tile_sum)
                    expert_indices_map_order = self.get_expert_indices_map_order(
                        offset,
                        module.num_experts,
                        module.d_model,
                        module.dim_feedforward,
                    )

                    for idx in expert_indices_map_order:
                        mapping, tiles_in = self.shape_to_mapping(
                            inp_shape=ffn1s[idx].weight.shape,
                            utilization=1 / module.num_experts,
                            # is_repeated=module.num_experts,
                        )
                        ffn1s[idx].set_mapping(mapping=mapping)

                        if self.map_strategy.split_ffn:
                            mapping, _ = self.shape_to_mapping(
                                inp_shape=ffn2s[idx].weight.shape,
                                utilization=1 / module.num_experts,
                            )
                            ffn2s[idx].set_mapping(mapping=mapping)
                        else:
                            mapping, _ = self.shape_to_mapping(
                                inp_shape=ffn2s[idx].weight.shape,
                                utilization=1 / module.num_experts,
                                tile_indices=tiles_in,
                            )
                            ffn2s[idx].set_mapping(mapping=mapping)
                else:
                    raise MappingException("Unknown mapping routine.")

        # check if every module was mapped
        for n, m in self.model.named_modules():
            if isinstance(m, Linear):
                assert (
                    hasattr(m, "mapping") and m.mapping is not None
                ), f"Forgot to map {n}"

    def shape_to_mapping(
        self,
        inp_shape: tuple[int],
        utilization,
        is_repeated=0,
        tile_indices=None,
        dry_run=False,
    ):
        """
        Function that maps the matrix with given shape to the accelerator.
        Args:
            inp_shape (tuple[int]): Shape of the matrix to be mapped.
            name (str): Name of the weight to be mapped.
            tile_indices (list[int], optional): List of tile indices that we
                want to use for mapping the shape. Defaults to None.
            dry_run (bool): If True, doesn't actually change any state but returns
                where it would have mapped the given input shape.
        Returns:
            tuple[dict, list[int]]: Tuple with dictionary that is mapped and
                a list of the tiles that are used for mapping.
        """
        tier_shape = self.accelerator.config.tier_shape
        assert (
            self.tile_sum < self.accelerator.config.tiers
        ).any(), "Network does not fit"
        # we change the state in a dry run, but restore it later on
        if dry_run:
            stored_tile_sum = deepcopy(self.tile_sum)
            stored_weighted_tile_sum = deepcopy(self.weighted_tile_sum)

        inp_rows, inp_cols = inp_shape
        tier_rows, tier_cols = tier_shape
        num_x = math.ceil(inp_rows / tier_rows)
        num_y = math.ceil(inp_cols / tier_cols)
        mapping, used_tiles = [], []
        tile_offset = 0
        for row_idx in range(num_x):
            row_mapping = []
            for col_idx in range(num_y):
                # what is the amount of weights we waste?
                rows_excess = max((row_idx + 1) * tier_rows - inp_rows, 0)
                cols_excess = max((col_idx + 1) * tier_cols - inp_cols, 0)
                used_rows = tier_rows - rows_excess
                used_cols = tier_cols - cols_excess
                tier_utilization = (used_rows * used_cols) / (tier_rows * tier_cols)

                # - Find the tile that has the least used tiers
                if tile_indices is None:
                    if self.map_strategy.strategy in [
                        Strategy.GREEDY_UTIL,
                        Strategy.MOE_SEPARATED,
                    ]:
                        # - We need to avoid replications in the used tiles. This avoids
                        # mapping chunks of a layer to the same tile.
                        sorted_weighted_sum = np.sort(self.weighted_tile_sum)
                        for min_val in np.unique(sorted_weighted_sum):
                            candidate_available_tiles = np.where(
                                self.weighted_tile_sum == min_val
                            )[0]
                            available_tiles = []
                            for candidate in candidate_available_tiles:
                                if (
                                    num_x * num_y > self.accelerator.config.tiles
                                    or (not candidate in used_tiles)
                                ) and (
                                    self.tile_sum[candidate] + is_repeated
                                    < self.accelerator.config.tiers
                                ):
                                    available_tiles.append(candidate)
                            if available_tiles != []:
                                break

                        if available_tiles == []:
                            # try without the constraint that we want to
                            # avoid using the same tiles.
                            for min_val in np.unique(sorted_weighted_sum):
                                candidate_available_tiles = np.where(
                                    self.weighted_tile_sum == min_val
                                )[0]
                                available_tiles = []
                                for candidate in candidate_available_tiles:
                                    if (
                                        self.tile_sum[candidate] + is_repeated
                                        < self.accelerator.config.tiers
                                    ):
                                        available_tiles.append(candidate)
                                if available_tiles != []:
                                    break

                    else:
                        available_tiles = np.where(
                            self.tile_sum == np.min(self.tile_sum)
                        )[0]

                    if self.map_strategy.strategy == Strategy.GREEDY_RANDOM:
                        # - Shuffle the available tiles. This is still greedy, but not ordered.
                        np.random.shuffle(available_tiles)
                        tile_idx = available_tiles[0]
                    else:
                        # - Just pick the first one
                        if (
                            len(available_tiles) == 0
                            and (
                                self.tile_sum + is_repeated
                                >= self.accelerator.config.tiers
                            ).all()
                        ):
                            raise MappingException(
                                "Mapping Error. Stacking MoE experts is not possible anymore."
                            )

                        tile_idx = available_tiles[0]
                else:
                    tile_idx = tile_indices[tile_offset % len(tile_indices)]
                    if self.tile_sum[tile_idx] >= self.accelerator.config.tiers:
                        # this tile doesn't work out
                        available_tiles = np.where(
                            self.tile_sum == np.min(self.tile_sum)
                        )[0]
                        tile_idx = available_tiles[0]

                tile_offset += 1
                used_tiles.append(tile_idx)
                num_tiers_used = self.tile_sum[tile_idx]
                assert (
                    num_tiers_used < self.accelerator.config.tiers
                ), "Network does not fit"
                row_mapping.append(
                    (
                        int(tile_idx),
                        int(num_tiers_used),
                        tier_utilization,
                        used_rows,
                        used_cols,
                    )
                )
                # - Mark tier used
                self.tile_sum[tile_idx] += 1
                self.weighted_tile_sum[tile_idx] += utilization

            if dry_run:
                self.tile_sum = stored_tile_sum
                self.weighted_tile_sum = stored_weighted_tile_sum
            mapping.append(row_mapping)

        return mapping, used_tiles

    def gen_module_pairs(self):
        """
        For every layer, turn the mapping into a partial function call of shape_to_mapping.
        """
        num_params = self.num_params()
        num_tiles_needed = self.num_tiles_required()

        print(
            f"Network has {num_params:,} parameters and requires {num_tiles_needed:,} tiles of available {self.accelerator.config.tiles}"
        )

        def get_modules(m):
            if isinstance(m, (Linear, MoELayer, Embedding)):
                return m
            else:
                modules = [get_modules(c) for c in m.children()]
            return modules

        self.module_pairs = list(flatten(get_modules(self.model)))

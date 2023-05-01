import torch
from torch import Tensor
from typing import List, Tuple

# Reference: https://github.com/pytorch/vision/blob/main/torchvision/ops/poolers.py

def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 160,
    canonical_level: int = 4,
    eps: float = 1e-6,
):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


def box_volume(boxes: Tensor) -> Tensor:
    return boxes[..., 3] * boxes[..., 4] * boxes[..., 5]


class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 160,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: Tensor) -> Tensor:
        """
        Args:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.pow(box_volume(boxlists), 1.0 / 3.0)
        
        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


def _setup_scales(
    scales: List[int], canonical_scale: int, canonical_level: int
) -> Tuple[List[float], LevelMapper]:

    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()

    map_levels = initLevelMapper(
        int(lvl_min),
        int(lvl_max),
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    return map_levels

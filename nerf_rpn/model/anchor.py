# 3D anchor generator and RPN head adapted from torchvision:
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/anchor_utils.py
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/rpn.py

from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import itertools


# 3D anchor generator for anchor-based methods
class AnchorGenerator3D(nn.Module):
    # type: (List[int], List[List[int]], int, int, Device) 
    '''
    Args:
        sizes: list of list of float. Each list corresponds to one level of features.
               Each number corresponds to a certain size of base anchor.
        aspect_ratios: 
        is_normlized: If it is False, the model will not normalize the value of 
                      aspect_ratios(i.e. set the volume of each base anchor as 1)


    The number of base anchors at each level is (permutation of ratios x length of size). 
    See num_anchors_per_location.


    Examples of input:
        anchor_sizes = ((1., 2., 3., 4., 5.), (6., 7., 8., 9., 10.))
        aspect_ratios = (((1., 1., 2.), (2., 2., 2.), (1., 2., 2.),),) * len(anchor_sizes)
        The number of base anchors at each level is 17 in this example.
    '''

    def __init__(self, sizes, aspect_ratios, is_normalized = False):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.is_normalized = is_normalized
        self.aspect_ratios_unique = []
        for size in self.aspect_ratios:
            cur_ratios = set()
            for ratio in size:
                cur_ratios.update(set(itertools.permutations(ratio)))
            
            self.aspect_ratios_unique.append(list(cur_ratios))

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios_unique)]

    def generate_anchors(self, scales, xyz_ratios, dtype=torch.float32, device="cpu"):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        w_ratios = []
        h_ratios = []
        d_ratios = []

        for ratio in xyz_ratios:

            P = list(itertools.permutations(ratio))
            P = torch.tensor(list(set(P)))

            if self.is_normalized:
                weight = torch.tensor(1.)
                for i in range(3):
                    weight *= ratio[i]
                weight = torch.pow(weight, 1. / 3.)
                P = P / weight

            w_ratios.append(P[:, 0])
            h_ratios.append(P[:, 1])
            d_ratios.append(P[:, 2])

        w_ratios = torch.cat(w_ratios, dim=0).to(device)
        h_ratios = torch.cat(h_ratios, dim=0).to(device)
        d_ratios = torch.cat(d_ratios, dim=0).to(device)

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        ds = (d_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, -ds, ws, hs, ds], dim=1) / 2
        return base_anchors.round()

    # we put our anchors in different group. One group has a specific size of anchor.
    def set_cell_anchors(self, dtype, device):
        cell_anchors = [
            self.generate_anchors(
                sizes,
                aspect_ratios,
                dtype,
                device
            )
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    # Since we use the pyramid feature, we let one size of feature correspond to one size of anchors
    def grid_anchors(self, grid_sizes, strides, dtype=torch.float32, device="cpu"):
        anchors = []

        for size, stride, base_anchors in zip(grid_sizes, strides, self.cell_anchors):
            grid_x, grid_y, grid_z = size
            stride_x, stride_y, stride_z = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, z_center, x_center, y_center, z_center]
            shifts_x = torch.arange(0, grid_x, dtype=torch.float32, device=device) * stride_x
            shifts_y = torch.arange(0, grid_y, dtype=torch.float32, device=device) * stride_y
            shifts_z = torch.arange(0, grid_z, dtype=torch.float32, device=device) * stride_z

            shift_x, shift_y, shift_z = torch.meshgrid(shifts_x, shifts_y, shifts_z, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shift_z = shift_z.reshape(-1)

            shifts = torch.stack((shift_x, shift_y, shift_z, shift_x, shift_y, shift_z), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, 6) + base_anchors.view(1, -1, 6)).reshape(-1, 6))

        return anchors

    def get_padding_masks(self, meshes, feature_maps, ori_sizes):
        '''
        Return a list of masks with the shape of (N, A, W, H, D), each corresponds to
        a level in the feature maps. True values in the masks mark for valid anchors 
        (not in the zero-padded regions).
        '''
        grid_sizes = list([feature_map.shape[-3:] for feature_map in feature_maps])
        mesh_size = meshes.shape[-3:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        
        strides = [[torch.tensor(mesh_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(mesh_size[1] // g[1], dtype=torch.int64, device=device),
                    torch.tensor(mesh_size[2] // g[2], dtype=torch.int64, device=device)] for g in grid_sizes]

        masks = []
        num_anchors_all = self.num_anchors_per_location()
        for size, stride, num_anchors in zip(grid_sizes, strides, num_anchors_all):
            masks_single_level = []
            stride = torch.tensor(stride)
            for ori_size in ori_sizes:
                ori_size = torch.tensor(ori_size)
                limits = torch.ceil(ori_size / stride).to(dtype=torch.int64, device=device)
                mask = torch.zeros((num_anchors, *size), dtype=torch.bool, device=device)
                mask[:, :limits[0], :limits[1], :limits[2]] = True
                masks_single_level.append(mask)

            masks.append(torch.stack(masks_single_level, dim=0))

        return masks

    def forward(self, meshes: Tensor, feature_maps: List[Tensor]):
        grid_sizes = list([feature_map.shape[-3:] for feature_map in feature_maps])
        # print(f'meshes: {meshes.shape}')
        mesh_size = meshes.shape[-3:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        strides = [[torch.tensor(mesh_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(mesh_size[1] // g[1], dtype=torch.int64, device=device),
                    torch.tensor(mesh_size[2] // g[2], dtype=torch.int64, device=device)] for g in grid_sizes]

        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors = []
        for _ in range(meshes.shape[0]):
            anchors_in_mesh = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_mesh)

        
        non_cat_anchors = anchors
        anchors = [torch.cat(anchors_per_mesh) for anchors_per_mesh in anchors]
        return anchors, non_cat_anchors


class RPNHead(nn.Module):
    """
    Outputs objectness logits and bounding box regression deltas.
    - If rotate=True, box_regression: [N, num_anchors, (dx, dy, dz, log(dw), log(dh), log(dd))]
    - If rotate=False, box_regression: [N, num_anchors, (dx, dy, dz, dw, dh, dd, da, db)]

    Check coder.AABBCoder or coder.MidpointOffsetCoder for the meaning of the elements in the tuple.
    """
    
    def __init__(self, in_channels, num_anchors, conv_depth=1, rotate=False):
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.append(nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1))
            convs.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*convs)
        self.cls_logits = nn.Conv3d(in_channels, num_anchors, kernel_size=1, stride=1)
        if not rotate:
            self.bbox_pred = nn.Conv3d(in_channels, num_anchors * 6, kernel_size=1, stride=1)
        else:
            self.bbox_pred = nn.Conv3d(in_channels, num_anchors * 8, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]            

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

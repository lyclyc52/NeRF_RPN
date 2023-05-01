from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .anchor import AnchorGenerator3D, RPNHead
from .rpn import RegionProposalNetwork

import numpy as np


def _default_anchorgen():
    anchor_sizes = ((8,), (16,), (32,), (64,),)
    aspect_ratios = (((1., 1., 1.), (1., 1., 2.), (1., 2., 2.), (1., 1., 3.), 
                    (1., 3., 3.)),) * len(anchor_sizes)
    return AnchorGenerator3D(anchor_sizes, aspect_ratios)


# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
class NeRFRegionProposalNetwork(nn.Module):
    """
    Implements RPN over NeRF.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W, D], one for each
    NeRF. Different tensors can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of tensors),
    containing:
        Two boxes options: 
        - AABB boxes: (``FloatTensor[N, 6]``): the ground-truth boxes in ``[x1, y1, z1, x2, y2, z2]`` format, 
          with ``0 <= x1 < x2 <= W``, ``0 <= y1 < y2 <= H``, and ``0 <= z1 < z2 <= D``.
        - OBB boxes: (``FloatTensor[N, 7]``): the ground-truth boxes in ``[x, y, z, w, h, d, theta]`` format.

    The model returns a Dict[Tensor] and two List[Tensor] during training, containing the classification 
    and regression losses for the RPN, the proposals and their scores.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
    """

    def __init__(
        self,
        backbone,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        iou_batch_size=16,
        rotated_bbox=False,
        reg_loss_type="smooth_l1",
        **kwargs,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        if not isinstance(rpn_anchor_generator, (AnchorGenerator3D, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            rpn_anchor_generator = _default_anchorgen()
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0], rotated_bbox)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
            iou_batch_size=iou_batch_size,
            rotated_bbox=rotated_bbox,
            reg_loss_type=reg_loss_type,
        )

        super().__init__()
        self.backbone = backbone
        self.rpn = rpn

    def transform(self, meshes, targets=None):
        """
        Do 0-padding to support different sizes of input meshes.
        Note: We do padding based on the each input batch, which means that the output of the same scene can 
            be different when the other scenes in the input batch are different
        Args:
            meshes (list[Tensor]): list of meshes to be transformed.
            targets (list[Tensor]): list of tensors containing the ground-truth boxes for each mesh.
        """
        if len(meshes) > 1:
            shapes = [mesh.shape for mesh in meshes]
            target_shape = np.max(shapes, axis=0)
            # print(f'Padding to {target_shape}')
            for i, mesh in enumerate(meshes):
                meshes[i] = F.pad(mesh, (0, target_shape[-1] - mesh.shape[-1], 0, target_shape[-2] - mesh.shape[-2], 
                                  0, target_shape[-3] - mesh.shape[-3]), mode="constant", value=0)

        return meshes, targets
    
    def check_bbox_degeneration(self, targets):
        if targets is not None:
            for target_idx, boxes in enumerate(targets):
                if boxes.shape[1] == 6:
                    degenerate_boxes = boxes[:, 3:] <= boxes[:, :3]
                else:
                    degenerate_boxes = boxes[:, 3:6] <= 0
                    
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height, width and depth."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

    def forward(self, meshes, targets=None, objectness_output_paths=None):
        """
        Args:
            meshes (list[Tensor]): meshes to be processed
            targets (list[Tensor])(optional): ground-truth boxes present in the scenes 
            objectness_output_paths (str)(optional): It can be activated through --output_voxel_scores 
                flag. See rpn_network/run_rpn.py for more details. 

        Returns:
            result (tuple(list[Tensor], dict[Tensor], list[Tensor])): the output from the model.
                The first element is a list of tensors, each of which contains the proposals of
                shape [N, 6] for each scene. The second element is a dictionary of losses. The third
                element is a list of tensors, each of which contains the scores for each proposal.

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for boxes in targets:
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and (boxes.shape[-1] == 6 or boxes.shape[-1] == 7),
                            f"Expected target boxes to be a tensor of shape [N, 6], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_mesh_sizes: List[Tuple[int, int, int]] = []
        for mesh in meshes:
            val = mesh.shape[-3:]
            torch._assert(
                len(val) == 3,
                f"expecting the last three dimensions of the Tensor to be W, H and D instead got {mesh.shape[-3:]}",
            )
            original_mesh_sizes.append((val[0], val[1], val[2]))

        # Transformation (0-padding) for batch size > 1
        meshes, targets = self.transform(meshes, targets)

        # Check for degenerate boxes (with negative side length)
        self.check_bbox_degeneration(targets)
        
        # Here we assume either the batch size is 1 or the grids are of the same size
        mesh_tensors = torch.stack(meshes, dim=0)
        features = list(self.backbone(mesh_tensors))

        proposals, level_index, proposal_losses, scores = self.rpn(
            mesh_tensors, features, original_mesh_sizes, targets, objectness_output_paths
        )

        return [features, proposals, level_index], proposal_losses, scores

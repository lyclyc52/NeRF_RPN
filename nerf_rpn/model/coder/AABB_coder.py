import torch
import math
from typing import List
from torch import Tensor
from .base_bbox_coder import BaseBBoxCoder

def encode_boxes_3d(reference_boxes: Tensor, proposals: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """

    # perform some unpacking to make it JIT-fusion friendly
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_z1 = proposals[:, 2].unsqueeze(1)
    proposals_x2 = proposals[:, 3].unsqueeze(1)
    proposals_y2 = proposals[:, 4].unsqueeze(1)
    proposals_z2 = proposals[:, 5].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_z1 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 3].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 4].unsqueeze(1)
    reference_boxes_z2 = reference_boxes[:, 5].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_depths = proposals_z2 - proposals_z1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights
    ex_ctr_z = proposals_z1 + 0.5 * ex_depths

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_depths = reference_boxes_z2 - reference_boxes_z1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights
    gt_ctr_z = reference_boxes_z1 + 0.5 * gt_depths

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dz = (gt_ctr_z - ex_ctr_z) / ex_depths
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)
    targets_dd = torch.log(gt_depths / ex_depths)

    targets = torch.cat((targets_dx, targets_dy, targets_dz, 
                         targets_dw, targets_dh, targets_dd), dim=1)
    return targets

class AABBCoder(BaseBBoxCoder):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self, bbox_xform_clip: float = math.log(2000.0)
    ) -> None:
        """
        Args:
            bbox_xform_clip (float)
        """
        self.bbox_xform_clip = bbox_xform_clip

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        targets = encode_boxes_3d(reference_boxes, proposals)

        return targets

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 3] - boxes[:, 0]
        heights = boxes[:, 4] - boxes[:, 1]
        depths = boxes[:, 5] - boxes[:, 2]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        ctr_z = boxes[:, 2] + 0.5 * depths

        dx = rel_codes[:, 0::6]
        dy = rel_codes[:, 1::6]
        dz = rel_codes[:, 2::6]
        dw = rel_codes[:, 3::6]
        dh = rel_codes[:, 4::6]
        dd = rel_codes[:, 5::6]

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        dd = torch.clamp(dd, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_ctr_z = dz * depths[:, None] + ctr_z[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_d = torch.exp(dd) * depths[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        c_to_c_d = torch.tensor(0.5, dtype=pred_ctr_z.dtype, device=pred_d.device) * pred_d

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_z - c_to_c_d
        pred_boxes4 = pred_ctr_x + c_to_c_w
        pred_boxes5 = pred_ctr_y + c_to_c_h
        pred_boxes6 = pred_ctr_z + c_to_c_d
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, 
                                  pred_boxes4, pred_boxes5, pred_boxes6), dim=2).flatten(1)
        return pred_boxes

    


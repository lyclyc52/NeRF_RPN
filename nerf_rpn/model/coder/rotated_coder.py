# ------------------------------------------
#
# This part of code refers to https://github.com/dingjiansw101/RoITransformer_DOTA
#
# ------------------------------------------

import torch
from torch import Tensor

from .base_bbox_coder import BaseBBoxCoder
import math

class RotatedCoder(BaseBBoxCoder):

    def __init__(self, bbox_xform_clip: float = math.log(2000.0)):

        self.bbox_xform_clip = torch.tensor(bbox_xform_clip)

    def encode_single(self, gt_rois: Tensor, ex_drois: Tensor):
        """
        Get deltas from anchors with respect to gt_bboxes.
        Args:
            ex_drois (Tensor): (N, 7), [x, y, z, w, h, d, theta]
            gt_rois (Tensor): (N, 7), [x, y, z, w, h, d, theta]
        Returns:
            deltas (Tensor): (N, 7), [dx, dy, dz, dw, dh, dd, dtheta]
        """
        # pdb.set_trace()
        gt_widths = gt_rois[:, 3]
        gt_heights = gt_rois[:, 4]
        gt_depths = gt_rois[:, 5]
        gt_angle = gt_rois[:, 6]

        ex_widths = ex_drois[:, 3]
        ex_heights = ex_drois[:, 4]
        ex_depths = ex_drois[:, 5]
        ex_angle = ex_drois[:, 6]

        coord = gt_rois[:, 0:3] - ex_drois[:, 0:3]
        
        
        targets_dx = (torch.cos(ex_drois[:, 6]) * coord[:, 0] + torch.sin(ex_drois[:, 6]) * coord[:, 1]) / ex_widths
        targets_dy = (-torch.sin(ex_drois[:, 6]) * coord[:, 0] + torch.cos(ex_drois[:, 6]) * coord[:, 1]) / ex_heights

        targets_dz = coord[:, 2] / ex_depths
        
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
        targets_dd = torch.log(gt_depths / ex_depths)
        
        # version 1
        targets_dangle = (gt_angle - ex_angle) / ( 2* torch.pi)
        targets = torch.stack((targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dd, targets_dangle), 1)

        return targets

    def decode_single(self, deltas: Tensor, ex_drois: Tensor):
        """
        Get OBBs from anchors and pred_deltas.
        Args:
            deltas (Tensor): (N, 8), [dx, dy, dz, dw, dh, dd, dtheta]
            ex_drois (Tensor): (N, 7), [x, y, z, w, h, d, theta]
        Returns:
            obboxes (Tensor): (N, 7), [x, y, z, w, h, d, theta]
        """
        assert deltas.size(0) == ex_drois.size(0)
        if torch.cuda.is_available():
            if deltas.get_device() != -1:
                self.bbox_xform_clip = self.bbox_xform_clip.to(deltas.get_device())
        else:
            self.bbox_xform_clip = self.bbox_xform_clip.cpu()
        
        ctr_x = ex_drois[:, 0]
        ctr_y = ex_drois[:, 1]
        ctr_z = ex_drois[:, 2]
        widths = ex_drois[:, 3]
        heights = ex_drois[:, 4]
        depths = ex_drois[:, 5]
        angles = ex_drois[:, 6]


        dx = deltas[:, 0::7]
        dy = deltas[:, 1::7]
        dz = deltas[:, 2::7]
        dw = deltas[:, 3::7]
        dh = deltas[:, 4::7]
        dd = deltas[:, 5::7]
        dangle = deltas[:, 6::7]
        
        dw = torch.min(dw, self.bbox_xform_clip)
        dh = torch.min(dh, self.bbox_xform_clip)
        dd = torch.min(dd, self.bbox_xform_clip)

   
        pred_ctr_x = dx * widths[:, None] * torch.cos(angles[:, None]) \
                   - dy * heights[:, None] * torch.sin(angles[:, None]) + ctr_x[:, None]
        pred_ctr_y = dx * widths[:, None] * torch.sin(angles[:, None]) + \
                     dy * heights[:, None] * torch.cos(angles[:, None]) + ctr_y[:, None]
        pred_ctr_z = dz * depths[:, None] + ctr_z[:, None]
        
        
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]
        pred_d = torch.exp(dd) * depths[:, None]


        pred_angle = (2 * torch.pi) * dangle + angles[:, None]
        pred_angle = pred_angle % torch.pi
        pred_angle[pred_angle > torch.pi / 2] = pred_angle[pred_angle > torch.pi / 2] - torch.pi

        pred_dboxes = torch.ones_like(deltas)

        pred_dboxes[:, 0::7] = pred_ctr_x
        pred_dboxes[:, 1::7] = pred_ctr_y
        pred_dboxes[:, 2::7] = pred_ctr_z
        pred_dboxes[:, 3::7] = pred_w
        pred_dboxes[:, 4::7] = pred_h
        pred_dboxes[:, 5::7] = pred_d
        pred_dboxes[:, 6::7] = pred_angle

        return pred_dboxes
        

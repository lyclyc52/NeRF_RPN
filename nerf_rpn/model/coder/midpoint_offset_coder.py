import numpy as np
import torch
from torch import Tensor

from .base_bbox_coder import BaseBBoxCoder
from .misc import obb2hbb, obb2poly, rectpoly2obb

# Reference: https://github.com/jbwang1997/OBBDetection/blob/87ffb063108d0e1a2eefd110be8f3713f60a6496/mmdet/core/bbox/coder/obb/midpoint_offset_coder.py
class MidpointOffsetCoder(BaseBBoxCoder):

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1., 1., 1., 1.)):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds

    def encode_single(self, gt_bboxes: Tensor, anchors: Tensor):
        """
        Get deltas from anchors with respect to gt_bboxes.
        Args:
            gt_bboxes (Tensor): (N, 7), [x, y, z, w, h, d, theta]
            anchors (Tensor): (N, 6), [xmin, ymin, zmin, xmax, ymax, zmax]
        Returns:
            deltas (Tensor): (N, 8), [dx, dy, dz, dw, dh, dd, da, db]

        """
        # print(gt_bboxes.shape)
        # print(anchors.shape)
        assert anchors.size(0) == gt_bboxes.size(0)
        encoded_bboxes = bbox2delta_sp(anchors, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode_single(self, pred_deltas: Tensor, anchors: Tensor, wh_ratio_clip=16 / 1000):
        """
        Get OBBs from anchors and pred_deltas.
        Args:
            pred_deltas (Tensor): (N, 8), [dx, dy, dz, dw, dh, dd, da, db]
            anchors (Tensor): (N, 6), [xmin, ymin, zmin, xmax, ymax, zmax]
            wh_ratio_clip (float): Clip ratio of width and height.
        Returns:
            obboxes (Tensor): (N, 7), [x, y, z, w, h, d, theta]
        """
        assert pred_deltas.size(0) == anchors.size(0)
        decoded_bboxes = delta_sp2bbox(anchors, pred_deltas, self.means, self.stds, wh_ratio_clip)
        return decoded_bboxes


# rotated bbox to regression target
def rbbox2delta_sp(proposals, gt,
                  means=(0., 0., 0., 0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1., 1., 1., 1.)):
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0::6] + proposals[..., 3::6]) * 0.5
    py = (proposals[..., 1::6] + proposals[..., 4::6]) * 0.5
    pz = (proposals[..., 2::6] + proposals[..., 5::6]) * 0.5
    pw = proposals[..., 3::6] - proposals[..., 0::6]
    ph = proposals[..., 4::6] - proposals[..., 1::6]
    pd = proposals[..., 5::6] - proposals[..., 2::6]
    # print(px.shape)

    # third dimension
    gz = gt[..., 2::7]
    gd = gt[..., 5::7]

    # 2D stuff.
    gt_2d = torch.cat([gt[..., 0:2], gt[..., 3:5], gt[..., 6::6]], dim=-1)
    hbb, poly = obb2hbb(gt_2d), obb2poly(gt_2d)
    gx = (hbb[..., 0::6] + hbb[..., 2::6]) * 0.5
    gy = (hbb[..., 1::6] + hbb[..., 3::6]) * 0.5
    gw = hbb[..., 2::6] - hbb[..., 0::6]
    gh = hbb[..., 3::6] - hbb[..., 1::6]

    x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
    y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
    x_max, _ = torch.max(x_coor, dim=1, keepdim=True)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor-y_min) > 0.1] = -1000
    ga, _ = torch.max(_x_coor, dim=1, keepdim=True)

    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor-x_max) > 0.1] = -1000
    gb, _ = torch.max(_y_coor, dim=1, keepdim=True)

    # 3d output
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dz = (gz - pz) / pd
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    dd = torch.log(gd / pd)
    da = (ga - gx) / gw
    db = (gb - gy) / gh
    deltas = torch.cat([dx, dy, dz, dw, dh, dd, da, db], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


# anchor to regression target
def bbox2delta_sp(proposals, gt,
                  means=(0., 0., 0., 0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1., 1., 1., 1.)):
    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0::6] + proposals[..., 3::6]) * 0.5
    py = (proposals[..., 1::6] + proposals[..., 4::6]) * 0.5
    pz = (proposals[..., 2::6] + proposals[..., 5::6]) * 0.5
    pw = proposals[..., 3::6] - proposals[..., 0::6]
    ph = proposals[..., 4::6] - proposals[..., 1::6]
    pd = proposals[..., 5::6] - proposals[..., 2::6]
    # print(px.shape)

    # third dimension
    gz = gt[..., 2::7]
    gd = gt[..., 5::7]

    # 2D stuff. TODO: learn what subfunction does
    gt_2d = torch.cat([gt[..., 0:2], gt[..., 3:5], gt[..., 6::6]], dim=-1)
    hbb, poly = obb2hbb(gt_2d), obb2poly(gt_2d)
    gx = (hbb[..., 0::6] + hbb[..., 2::6]) * 0.5
    gy = (hbb[..., 1::6] + hbb[..., 3::6]) * 0.5
    gw = hbb[..., 2::6] - hbb[..., 0::6]
    gh = hbb[..., 3::6] - hbb[..., 1::6]

    x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
    y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
    x_max, _ = torch.max(x_coor, dim=1, keepdim=True)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor-y_min) > 0.1] = -1000
    ga, _ = torch.max(_x_coor, dim=1, keepdim=True)

    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor-x_max) > 0.1] = -1000
    gb, _ = torch.max(_y_coor, dim=1, keepdim=True)

    # 3d output
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dz = (gz - pz) / pd
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    dd = torch.log(gd / pd)
    da = (ga - gx) / gw
    db = (gb - gy) / gh
    deltas = torch.cat([dx, dy, dz, dw, dh, dd, da, db], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

def delta_sp2bbox(rois: Tensor, deltas: Tensor,
                  means=(0., 0., 0., 0., 0., 0., 0., 0.),
                  stds=(1., 1., 1., 1., 1., 1., 1., 1.),
                  wh_ratio_clip=16 / 1000):

    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 8)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 8)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::8]
    dy = denorm_deltas[:, 1::8]
    dz = denorm_deltas[:, 2::8]
    dw = denorm_deltas[:, 3::8]
    dh = denorm_deltas[:, 4::8]
    dd = denorm_deltas[:, 5::8]
    da = denorm_deltas[:, 6::8]
    db = denorm_deltas[:, 7::8]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    dd = dd.clamp(min=-max_ratio, max=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 4]) * 0.5).unsqueeze(1).expand_as(dy)
    pz = ((rois[:, 2] + rois[:, 5]) * 0.5).unsqueeze(1).expand_as(dz)
    # Compute width/height of each roi
    pw = (rois[:, 3] - rois[:, 0]).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 4] - rois[:, 1]).unsqueeze(1).expand_as(dh)
    pd = (rois[:, 5] - rois[:, 2]).unsqueeze(1).expand_as(dd)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gd = pd * dd.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    gz = pz + pd * dz

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    da = da.clamp(min=-0.5, max=0.5)
    db = db.clamp(min=-0.5, max=0.5)
    ga = gx + da * gw
    _ga = gx - da * gw
    gb = gy + db * gh
    _gb = gy - db * gh
    polys = torch.stack([ga, y1, x2, gb, _ga, y2, x1, _gb], dim=-1)

    # rectangulate polybbox
    center = torch.stack([gx, gy, gx, gy, gx, gy, gx, gy], dim=-1)
    center_polys = polys - center
    diag_len = torch.sqrt(torch.square(center_polys[..., 0::2]) + torch.square(center_polys[..., 1::2]))
    max_diag_len, _ = torch.max(diag_len, dim=-1, keepdim=True)
    diag_scale_factor: Tensor = max_diag_len / diag_len
    center_polys = center_polys * diag_scale_factor.repeat_interleave(2, dim=-1)
    rectpolys = center_polys + center

    # compute obbox
    obboxes_2d = rectpoly2obb(rectpolys).flatten(-2) # (x, y, w, h, theta)
    obboxes = torch.cat([obboxes_2d[..., 0:2], gz, obboxes_2d[..., 2:4], gd, obboxes_2d[..., 4::5]], dim=-1) # (x, y, z, w, h, d, theta)
    
    return obboxes
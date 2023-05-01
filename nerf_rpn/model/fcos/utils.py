import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from ..rotated_iou.oriented_iou_loss import cal_iou_3d, box2corners_th


def decode_fcos_obb(locations, box_regression):
    """
    Arguments:
        locations (tensor[N, 3]): locations of the anchors
        box_regression (tensor[N, 8]): predicted deltas using midpoint offsets
    Returns:
        boxes (tensor[N, 7]): predicted boxes
    """

    assert box_regression.shape[1] == 8, "box_regression for OBB should have 8 offsets"

    x0 = locations[:, 0] - box_regression[:, 0]
    y0 = locations[:, 1] - box_regression[:, 1]
    z0 = locations[:, 2] - box_regression[:, 2]
    x1 = locations[:, 0] + box_regression[:, 3]
    y1 = locations[:, 1] + box_regression[:, 4]
    z1 = locations[:, 2] + box_regression[:, 5]
    vx = (x1 + x0) / 2 + box_regression[:, 6] * (x1 - x0)
    vy = (y1 + y0) / 2 + box_regression[:, 7] * (y1 - y0)

    vx = torch.clamp(vx, min=x0, max=x1)
    vy = torch.clamp(vy, min=y0, max=y1)

    # Locate the centers of the boxes
    centers = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], dim=1)

    # Locate two vertices of the OBBs
    v0 = torch.stack([vx, y1], dim=1)
    v1 = torch.stack([x1, vy], dim=1)

    # Rectangularize the OBBs
    v0 = v0 - centers[:, :2]
    v1 = v1 - centers[:, :2]
    d0 = torch.norm(v0, dim=1)
    d1 = torch.norm(v1, dim=1)
    dmax = torch.max(d0, d1)
    v0 = v0 / (d0[:, None] + 1e-7) * dmax[:, None] + centers[:, :2]
    v1 = v1 / (d1[:, None] + 1e-7) * dmax[:, None] + centers[:, :2]

    # Compute the lengths of the sides of the OBBs
    l = torch.norm(v0 - v1, dim=1)
    w = torch.norm((v0 + v1) / 2 - centers[:, :2], dim=1) * 2
    h = z1 - z0

    # Compute the angles of the sides of the OBBs
    mid = (v0 + v1) / 2 - centers[:, :2]
    mid[(mid[:, 0] == 0) & (mid[:, 1] == 0), 0] = 1e-7
    theta = torch.atan2(mid[:, 1], mid[:, 0])

    return torch.stack([centers[:, 0], centers[:, 1], centers[:, 2], w, l, h, theta], dim=1)


def encode_fcos_obb(locations, boxes):
    assert boxes.shape[1] == 7, "input OBB should have 7 parameters"
    assert boxes.shape[0] == locations.shape[0], "number of boxes should be equal to number of locations"

    # Extract 2D projections of the OBBs
    proj = boxes[..., [0, 1, 3, 4, 6]]
    corners = box2corners_th(proj.unsqueeze(0)).squeeze(0)

    # Find AABB of the OBBs
    xs = corners[:, :, 0]
    ys = corners[:, :, 1]

    xmax = xs.max(dim=1)[0]
    ymax = ys.max(dim=1)[0]
    xmin = xs.min(dim=1)[0]
    ymin = ys.min(dim=1)[0]

    x0 = locations[:, 0] - xmin
    y0 = locations[:, 1] - ymin
    z0 = locations[:, 2] - (boxes[:, 2] - boxes[:, 5] / 2)
    x1 = xmax - locations[:, 0]
    y1 = ymax - locations[:, 1]
    z1 = (boxes[:, 2] + boxes[:, 5] / 2) - locations[:, 2]

    # Corner case: OBB close to AABB
    xt = xs.clone()
    yt = ys.clone()
    xt[ymax.unsqueeze(1) - ys > 0.1] = -1e6
    yt[xmax.unsqueeze(1) - xs > 0.1] = 1e6

    vx = xt.max(dim=1)[0]
    vy = yt.min(dim=1)[0]

    # Use AABB instead of OBB when theta is too small to be stable
    ids = torch.isclose(vx, xmax) & torch.isclose(vy, ymin)
    vx[ids] = xmax[ids]
    vy[ids] = ymin[ids]

    alpha = (vx - boxes[:, 0]) / (xmax - xmin)
    beta = (vy - boxes[:, 1]) / (ymax - ymin)

    return torch.stack([x0, y0, z0, x1, y1, z1, alpha, beta], dim=1)


# Reference: https://blog.csdn.net/hxxjxw/article/details/122629725
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    keep = []
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        # Pick the largest box and remove any overlap
        i = idxs[0]
        keep.append(i)
        if idxs.numel() == 1:
            break
        # Compute IoU of the picked box with the rest
        iou = box_iou_3d(boxes[i].unsqueeze(0), boxes[idxs[1:]]).squeeze()
        # Remove boxes with an IoU above the threshold
        idxs = idxs[1:][iou <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 6]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with ``0 <= x1 < x2``,
            ``0 <= y1 < y2`` and ``0 <= z1 < z2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for class_id in torch.unique(idxs):
        curr_indices = torch.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Remove boxes which contains at least one side smaller than min_size.

    Args:
        boxes (Tensor[N, 6] or Tensor[N, 7]): boxes in 
            ``(x1, y1, z1, x2, y2, z2)`` format with ``0 <= x1 < x2``, 
            ``0 <= y1 < y2`` and ``0 <= z1 < z2``, or in 
            ``(x, y, z, w, l, h, theta)`` format.
        min_size (float): minimum size

    Returns:
        Tensor[K]: indices of the boxes that have all sides
        larger than min_size
    """
    if boxes.shape[1] == 6:
        ws, ls, hs = boxes[:, 3] - boxes[:, 0], boxes[:, 4] - boxes[:, 1], boxes[:, 5] - boxes[:, 2]
    else:
        ws, ls, hs = boxes[:, 3], boxes[:, 4], boxes[:, 5]

    keep = (ws >= min_size) & (ls >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_mesh(boxes: Tensor, size: Tuple[int, int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside a mesh of size `size`.

    Args:
        boxes (Tensor[N, 6]): boxes in ``(x1, y1, z1, x2, y2, z2)`` format
            with ``0 <= x1 < x2``, ``0 <= y1 < y2`` and ``0 <= z1 < z2``.
        size (Tuple[width, height, depth]): size of the mesh

    Returns:
        Tensor[N, 6]: clipped boxes
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::3]
    boxes_y = boxes[..., 1::3]
    boxes_z = boxes[..., 2::3]
    width, height, depth = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)
    boxes_z = boxes_z.clamp(min=0, max=depth)

    clipped_boxes = torch.stack((boxes_x, boxes_y, boxes_z), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


@torch.no_grad()
def batched_box_iou(boxes1: Tensor, boxes2: Tensor, batch_size=16) -> Tensor:
    '''
    Batchify the box_iou_3d function to avoid OOM issues.
    '''
    ious = []
    num_batches = (boxes1.shape[0] + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        if end_idx > boxes1.shape[0]:
            end_idx = boxes1.shape[0]
        ious.append(box_iou_3d(boxes1[start_idx:end_idx], boxes2))

    return torch.cat(ious, dim=0)


def box_iou_3d(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be either in ``(x1, y1, z1, x2, y2, z2)`` format 
    with ``0 <= x1 < x2``, ``0 <= y1 < y2``, and ``0 <= z1 < z2``, or in
    ``(x, y, z, w, l, h, theta)``.

    Args:
        boxes1 (Tensor[N, 6] or Tensor[N, 7]): first set of boxes
        boxes2 (Tensor[M, 6] or Tensor[N, 7]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    assert boxes1.shape[1] == 6 or boxes1.shape[1] == 7, "boxes should have 6 or 7 elements"
    assert boxes1.shape[1] == boxes2.shape[1], "boxes1 and boxes2 should have the same number of elements"

    if boxes1.shape[1] == 6:
        inter, union = _box_inter_union_3d(boxes1, boxes2)
        iou = inter / union
    else:
        boxes1_rep = boxes1.unsqueeze(1).repeat(1, boxes2.size(0), 1)
        boxes2_rep = boxes2.unsqueeze(0).repeat(boxes1.size(0), 1, 1)
        iou = cal_iou_3d(boxes1_rep.cuda(), boxes2_rep.cuda()).to(dtype=torch.float32, device=boxes1.device)

    return iou


def box_volume(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, z1, x2, y2, z2) coordinates.

    Args:
        boxes (Tensor[N, 6]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, z1, x2, y2, z2) format with
            ``0 <= x1 < x2``, ``0 <= y1 < y2``, and ``0 <= z1 < z2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union_3d(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    volume1 = box_volume(boxes1)
    volume2 = box_volume(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    whd = _upcast(rb - lt).clamp(min=0)  # [N,M,3]
    inter = whd[:, :, 0] * whd[:, :, 1] * whd[:, :, 2]  # [N,M]

    union = volume1[:, None] + volume2 - inter

    return inter, union


def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """
    def normalize(x, axis=-1, order=2):
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,

    if at is None:
        at = np.zeros_like(camera_position)
    else:
        at = np.array(at)
    if up is None:
        up = np.zeros_like(camera_position)
        up[2] = -1
    else:
        up = np.array(up)

    z_axis = normalize(camera_position - at)[0]
    x_axis = normalize(np.cross(up, z_axis))[0]
    y_axis = normalize(np.cross(z_axis, x_axis))[0]

    R = np.concatenate([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R


def c2w_from_loc_and_at(cam_pos, at, up=(0, 0, 1)):
    """ Convert camera location and direction to camera2world matrix. """
    c2w = np.eye(4)
    cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
    c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
    return c2w


def get_w2cs(res : int = 160):
    """ Get world2camera matrices. """
    centroid = np.array([res/2]*3)
    positions = np.array([[res, res, res], [res, -res, res], [-res, res, res], [-res, -res, res]]) + centroid
    w2cs = [torch.Tensor(np.linalg.inv(c2w_from_loc_and_at(position, centroid))) for position in positions]
    if torch.cuda.is_available():
        w2cs = [w2c.cuda() for w2c in w2cs]
    return w2cs


def project(intrinsic_mat,  # [3x3]
            pose_mat,  # [4x4], world coord -> camera coord
            box_coords  # [Nx4] (x,y,z,1)
            ):
    """ Project 3D homogeneous coords to 2D coords.  """
    #-- From world space to camera space.  
    box_coords_cam = torch.matmul(pose_mat, torch.transpose(box_coords, 0, 1).float())  # [4x8]
    # box_coords_cam[:3, :] = box_coords_cam[:3, :] / box_coords_cam[3, :]
    
    #-- From camera space to picture space. 
    box_coords_pic = torch.matmul(intrinsic_mat, box_coords_cam[:3, :])  # [3x8]
    final_coords_raw = box_coords_pic[:2, :] / box_coords_pic[2, :]
    final_coords = torch.transpose(final_coords_raw, 0, 1)

    return final_coords


def obb2points_3d(obboxes):
    """ preprocessing for 2d loss """
    # [x, y, z, w, l, h, theta]
    center, w, l, h, theta = torch.split(obboxes, [3, 1, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    vector = torch.cat([w/2 * Cos -l/2 * Sin, w/2 * Sin + l/2 * Cos, h/2], dim=-1)
    return torch.cat([center-vector, center+vector], dim=0)


def pt(obj):    
    if type(obj) is list or type(obj) is tuple:
        shape_list = []
        for i in obj:
            shape_list.append(pt(i))
        return shape_list
    else:
        return obj.shape
        
        
def print_shape(obj):
    shape = pt(obj)
    print(shape)
    
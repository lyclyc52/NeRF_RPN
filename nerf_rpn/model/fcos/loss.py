"""
This file contains specific functions for computing losses of FCOS
Reference:
https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/modeling/rpn/fcos/loss.py

Modified to support 3D FCOS.
Note that the original FCOS implementation have its y axis pointing downwards.
We keep using the original FCOS implementation for NeRF RPN.
Therefore, in this file, 'left', 'top', and 'front' always refer to distance from
the center location of the object to the SMALLER boundary of the bounding box.

file
"""

import torch
from torch import nn
from torch.functional import F
from torchvision.ops import sigmoid_focal_loss
from .utils import decode_fcos_obb, encode_fcos_obb, get_w2cs, obb2points_3d, project
from ..rotated_iou.oriented_iou_loss import cal_iou_3d, cal_giou_3d, cal_diou_3d, box2corners_th


INF = 100000000


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def permute_and_flatten(layer, N, A, C, W, H, D):
    '''
    Permute and flatten from (N, A*C, W, H, D) to (N, W*H*D*A, C)
    '''
    layer = layer.view(N, -1, C, W, H, D)
    layer = layer.permute(0, 3, 4, 5, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, W, H, D = box_cls_per_level.shape
        Ax6 = box_regression_per_level.shape[1]
        A = Ax6 // 6
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, W, H, D
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 6, W, H, D
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 6)
    return box_cls, box_regression


class IOULoss(nn.Module):
    '''
    Modifed to support 3D IOU loss
    '''
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_front = pred[:, 2]
        pred_right = pred[:, 3]
        pred_bottom = pred[:, 4]
        pred_back = pred[:, 5]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_front = target[:, 2]
        target_right = target[:, 3]
        target_bottom = target[:, 4]
        target_back = target[:, 5]

        target_volume = (target_left + target_right) * \
                        (target_top + target_bottom) * \
                        (target_front + target_back)
        pred_volume = (pred_left + pred_right) * \
                      (pred_top + pred_bottom) * \
                      (pred_front + pred_back)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        d_intersect = torch.min(pred_front, target_front) + torch.min(pred_back, target_back)
        g_d_intersect = torch.max(pred_front, target_front) + torch.max(pred_back, target_back)

        ac_uion = g_w_intersect * g_h_intersect * g_d_intersect + 1e-7
        volume_intersect = w_intersect * h_intersect * d_intersect
        volume_union = target_volume + pred_volume - volume_intersect
        ious = (volume_intersect + 1.0) / (volume_union + 1.0)
        gious = ious - (ac_uion - volume_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class RotatedIOULoss(nn.Module):
    '''
    3D IoU loss for OBB
    '''
    def __init__(self, loss_type="iou"):
        super(RotatedIOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        dummy_loc = torch.zeros(pred.shape[0], 3, device=pred.device)

        pred_boxes = decode_fcos_obb(dummy_loc, pred)
        target_boxes = decode_fcos_obb(dummy_loc, target)

        if self.loss_type == 'iou' or self.loss_type == 'linear_iou':
            ious, _, _, _, unions = cal_iou_3d(pred_boxes.unsqueeze(0), target_boxes.unsqueeze(0), verbose=True)

            volume_intersect = ious * unions        
            ious = (volume_intersect + 1.0) / (unions + 1.0)

            if self.loss_type == 'iou':
                losses = -torch.log(ious)
            elif self.loss_type == 'linear_iou':
                losses = 1 - ious

        elif self.loss_type == 'giou':
            losses, _, _ = cal_giou_3d(pred_boxes.unsqueeze(0), target_boxes.unsqueeze(0))
        elif self.loss_type == 'diou':
            losses, _ = cal_diou_3d(pred_boxes.unsqueeze(0), target_boxes.unsqueeze(0))
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    Adapted for 3D case.
    """

    def __init__(self, fpn_strides, center_sampling_radius, iou_loss_type, norm_reg_targets, 
                 world_size, use_obb, use_additional_l1_loss, proj2d_loss_weight=0.0):
        self.cls_loss_func = sigmoid_focal_loss
        self.fpn_strides = fpn_strides
        self.center_sampling_radius = center_sampling_radius
        self.iou_loss_type = iou_loss_type
        self.norm_reg_targets = norm_reg_targets
        self.world_size = world_size
        self.use_obb = use_obb
        self.use_additional_l1_loss = use_additional_l1_loss
        self.proj2d_loss_weight = proj2d_loss_weight

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        
        if self.iou_loss_type != 'smooth_l1':
            self.box_reg_loss_func = IOULoss(self.iou_loss_type) if not use_obb else RotatedIOULoss(self.iou_loss_type)
        else:
            self.box_reg_loss_func = nn.SmoothL1Loss(reduction='none')
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.additional_l1_loss_func = nn.SmoothL1Loss(reduction="none")

    def reduce_sum(self, tensor):
        if self.world_size <= 1:
            return tensor
        import torch.distributed as dist
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, gt_zs, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 6)
        center_x = (gt[..., 0] + gt[..., 3]) / 2
        center_y = (gt[..., 1] + gt[..., 4]) / 2
        center_z = (gt[..., 2] + gt[..., 5]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if num_gts == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            zmin = center_z[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            zmax = center_z[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                zmin > gt[beg:end, :, 2], zmin, gt[beg:end, :, 2]
            )
            center_gt[beg:end, :, 3] = torch.where(
                xmax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], xmax
            )
            center_gt[beg:end, :, 4] = torch.where(
                ymax > gt[beg:end, :, 4],
                gt[beg:end, :, 4], ymax
            )
            center_gt[beg:end, :, 5] = torch.where(
                zmax > gt[beg:end, :, 5],
                gt[beg:end, :, 5], zmax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 3] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 4] - gt_ys[:, None]
        front = gt_zs[:, None] - center_gt[..., 2]
        back = center_gt[..., 5] - gt_zs[:, None]
        center_bbox = torch.stack((left, top, right, bottom, front, back), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 16],
            [16, 32],
            [32, 64],
            [64, INF]
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)

        if self.use_obb:
            labels, reg_targets = self.compute_targets_for_locations_obb(
                points_all_level, targets, expanded_object_sizes_of_interest
            )
        else:
            labels, reg_targets = self.compute_targets_for_locations(
                points_all_level, targets, expanded_object_sizes_of_interest
            )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level[..., :6] = reg_targets_per_level[..., :6] / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations_obb(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys, zs = locations[:, 0], locations[:, 1], locations[:, 2]

        for bboxes in targets:
            # No gt bboxes
            if bboxes.shape[0] == 0:
                labels.append(torch.zeros(locations.shape[0], device=locations.device))
                reg_targets.append(torch.zeros((locations.shape[0], 8), device=locations.device))
                continue

            # (#loc, #gt, 8)
            targets_per_loc = [encode_fcos_obb(locations, bboxes[i:i+1, :].expand(locations.shape[0], -1)) \
                for i in range(bboxes.shape[0])]
            reg_targets_per_im = torch.stack(targets_per_loc, dim=1)

            proj = bboxes[..., [0, 1, 3, 4, 6]]
            corners = box2corners_th(proj.unsqueeze(0)).squeeze(0)  # [n, 4, 2]
            aabbs = torch.cat([corners.min(dim=1)[0], bboxes[:, 2:3] - bboxes[:, 5:6] / 2,
                               corners.max(dim=1)[0], bboxes[:, 2:3] + bboxes[:, 5:6] / 2], dim=1)

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    aabbs,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys, zs,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im[..., :6].min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im[..., :6].max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            volumes = (aabbs[:, 3] - aabbs[:, 0]) * (aabbs[:, 4] - aabbs[:, 1]) * (aabbs[:, 5] - aabbs[:, 2])
            locations_to_gt_area = volumes[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

            # We are doing binary classification for RPN so far, so we can hard code that the labels are 1
            labels_per_im = torch.ones(locations.shape[0], device=locations.device)
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys, zs = locations[:, 0], locations[:, 1], locations[:, 2]

        for bboxes in targets:
            # No gt bboxes
            if bboxes.shape[0] == 0:
                labels.append(torch.zeros(locations.shape[0], device=locations.device))
                reg_targets.append(torch.zeros((locations.shape[0], 6), device=locations.device))
                continue

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            f = zs[:, None] - bboxes[:, 2][None]
            r = bboxes[:, 3][None] - xs[:, None]
            b = bboxes[:, 4][None] - ys[:, None]    # bottom
            ba = bboxes[:, 5][None] - zs[:, None]   # back
            reg_targets_per_im = torch.stack([l, t, f, r, b, ba], dim=2)

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys, zs,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            volumes = (bboxes[:, 3] - bboxes[:, 0]) * (bboxes[:, 4] - bboxes[:, 1]) * (bboxes[:, 5] - bboxes[:, 2])
            locations_to_gt_area = volumes[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

            # We are doing binary classification for RPN so far, so we can hard code that the labels are 1
            labels_per_im = torch.ones(locations.shape[0], device=locations.device)
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 3]]
        top_bottom = reg_targets[:, [1, 4]]
        front_back = reg_targets[:, [2, 5]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]) * \
                     (front_back.min(dim=-1)[0] / front_back.max(dim=-1)[0])
        return torch.sqrt(centerness)
    
    def compute_2d_projection_loss(self, box_reg, reg_targets, weights):
        # 2d reg loss
        w, h, fx, fy = 640, 480, 600, 600
        K = torch.tensor([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=torch.float32, device=box_reg.device)
        pose_list = get_w2cs(res=160)
        
        dummy_loc = torch.zeros(box_reg.shape[0], 3, device=box_reg.device)
        pred_boxes = decode_fcos_obb(dummy_loc, box_reg)
        target_boxes = decode_fcos_obb(dummy_loc, reg_targets)
        
        pred_boxes = obb2points_3d(pred_boxes).to(box_reg.device)
        target_boxes = obb2points_3d(target_boxes).to(box_reg.device)

        pred_boxes = torch.cat([pred_boxes, torch.ones(pred_boxes.shape[0], 1, device=box_reg.device)], dim=1)
        target_boxes = torch.cat([target_boxes, torch.ones(target_boxes.shape[0], 1, device=box_reg.device)], dim=1)
            
        pred_boxes_2d, target_boxes_2d = [], []
        for pose in pose_list:
            pred_boxes_2d.append(project(K, pose, pred_boxes))
            target_boxes_2d.append(project(K, pose, target_boxes))
            
        pred_boxes_2d = torch.cat(pred_boxes_2d, dim=0)
        target_boxes_2d = torch.cat(target_boxes_2d, dim=0)
        box_reg_loss_2d = F.smooth_l1_loss(pred_boxes_2d, target_boxes_2d, beta=1 / 9, reduction="none") / 160
        
        factor = box_reg_loss_2d.shape[0] // weights.shape[0]
        weights = weights[:, None].repeat(factor, 1)
        
        return (box_reg_loss_2d * weights).sum() / (factor * box_reg_loss_2d.shape[1])

    def __call__(self, locations, box_cls, box_regression, centerness, targets, padding_masks):
        """
        Arguments:
            locations (list[Tensor])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[Tensor])
            padding_masks (list[Tensor])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        box_reg_dim = 8 if self.use_obb else 6
        assert num_classes == 1 # for RPN only
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 4, 1).reshape(-1))   # binary classification
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 4, 1).reshape(-1, box_reg_dim))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, box_reg_dim))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        if padding_masks is not None:
            # Mask out locations that are in the padding regions.
            masks_flatten = [mask.reshape(-1) for mask in padding_masks]
            masks_flatten = torch.cat(masks_flatten, dim=0)

            box_cls_flatten = box_cls_flatten[masks_flatten]
            box_regression_flatten = box_regression_flatten[masks_flatten]
            centerness_flatten = centerness_flatten[masks_flatten]
            labels_flatten = labels_flatten[masks_flatten]
            reg_targets_flatten = reg_targets_flatten[masks_flatten]

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = self.world_size
        # sync num_pos from all gpus
        total_num_pos = self.reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten,
            reduction='sum'
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                self.reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            if self.iou_loss_type != 'smooth_l1':
                reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets_flatten,
                    centerness_targets
                ) / sum_centerness_targets_avg_per_gpu
            else:
                reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets_flatten,
                ) * centerness_targets.unsqueeze(1)
                reg_loss = reg_loss.sum() / sum_centerness_targets_avg_per_gpu
            
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu

            if self.use_obb and self.use_additional_l1_loss and self.iou_loss_type != 'smooth_l1':
                # additional smooth L1 loss for OBB midpoint offsets
                additional_l1_loss = self.additional_l1_loss_func(
                    box_regression_flatten[:, 6:],
                    reg_targets_flatten[:, 6:],
                ) * centerness_targets.unsqueeze(-1)
                additional_l1_loss = additional_l1_loss.sum() / sum_centerness_targets_avg_per_gpu
                reg_loss += additional_l1_loss
                
            if self.use_obb and self.proj2d_loss_weight > 0:
                proj2d_loss = self.compute_2d_projection_loss(
                    box_regression_flatten,
                    reg_targets_flatten,
                    centerness_targets
                ) / sum_centerness_targets_avg_per_gpu
                reg_loss += proj2d_loss * self.proj2d_loss_weight
        else:
            reg_loss = box_regression_flatten.sum()
            self.reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss

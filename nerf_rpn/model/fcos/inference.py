#
# Modified from https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/modeling/rpn/fcos/inference.py
# 3D version of FCOS inference
#


import torch
from .utils import clip_boxes_to_mesh, remove_small_boxes, nms, decode_fcos_obb, print_shape


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    Modified for 3D bounding boxes.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False,
        use_obb=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes  # Unused
        self.bbox_aug_enabled = bbox_aug_enabled
        self.use_obb = use_obb

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            grid_sizes, padding_masks):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, W, H, D
            box_regression: tensor of size N, A * 6 (A*8 if OBB), W, H, D
        """
        N, C, W, H, D = box_cls.shape

        box_reg_dim = 8 if self.use_obb else 6

        # put in the same format as locations
        box_cls = box_cls.view(N, C, W, H, D).permute(0, 2, 3, 4, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, box_reg_dim, W, H, D).permute(0, 2, 3, 4, 1)
        box_regression = box_regression.reshape(N, -1, box_reg_dim)
        centerness = centerness.view(N, 1, W, H, D).permute(0, 2, 3, 4, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        if padding_masks is not None:
            # Mask out padding areas by setting scores to be -1e5
            # box_cls already copied in previous sigmoid() call
            box_cls[~padding_masks] = -1e5

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        det_all = []
        scores_all = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1    # Always be one in RPN

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            if not self.use_obb:
                detections = torch.stack([
                    per_locations[:, 0] - per_box_regression[:, 0],
                    per_locations[:, 1] - per_box_regression[:, 1],
                    per_locations[:, 2] - per_box_regression[:, 2],
                    per_locations[:, 0] + per_box_regression[:, 3],
                    per_locations[:, 1] + per_box_regression[:, 4],
                    per_locations[:, 2] + per_box_regression[:, 5],
                ], dim=1)

                w, l, h = grid_sizes[i]
                detections = clip_boxes_to_mesh(detections, (w, l, h))
            else:
                # TODO: clip OBB to mesh
                detections = decode_fcos_obb(per_locations, per_box_regression)

            keep = remove_small_boxes(detections, self.min_size)
            detections = detections[keep]
            scores = torch.sqrt(per_box_cls[keep])

            det_all.append(detections)
            scores_all.append(scores)

        return det_all, scores_all

    def forward(self, locations, box_cls, box_regression, centerness, grid_sizes, padding_masks):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            grid_sizes: list[(w, h, d)]
            padding_masks: list[tensor]
        Returns:
            (boxes, scores) (tuple[list[tensor], list[tensor]]): the post-processed anchors, 
                after applying box decoding and NMS
        """
        boxes_all = []
        scores_all = []
        for level, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):

            masks_per_level = padding_masks[level] if padding_masks is not None else None
            boxes, scores = self.forward_for_single_feature_map(l, o, b, c, grid_sizes, masks_per_level)
            
            level_indinces = [torch.zeros(item.shape[:-1]).cuda() + level for item in boxes]

            boxes = [torch.cat([level_indinces[ind][..., None], boxes[ind]], dim = -1) for ind in range(len(boxes))]
            boxes_all.append(boxes)
            scores_all.append(scores)

        # Concatenate detections from different feature maps of the same scene together
        boxes_all = [torch.cat(x, dim=0) for x in zip(*boxes_all)]
        scores_all = [torch.cat(x, dim=0) for x in zip(*scores_all)]
        if not self.bbox_aug_enabled:
            return self.select_over_all_levels(boxes_all, scores_all)

        return boxes_all, scores_all

    def select_over_all_levels(self, boxes, scores):
        num_scenes = len(boxes)
        boxes_results = []
        scores_results = []
        indices_results = []
        box_indices = [item[..., 0] for item in boxes]
        boxes = [item[..., 1:] for item in boxes]
        for i in range(num_scenes):
            # Single class NMS, for RPN only
            keep = nms(boxes[i], scores[i], self.nms_thresh)
            number_of_detections = keep.numel()

            boxes_keep = boxes[i][keep]
            scores_keep = scores[i][keep]
            box_indices_keep = box_indices[i][keep]
            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                score_thresh, _ = torch.kthvalue(
                    scores_keep.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = scores_keep >= score_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                boxes_keep = boxes_keep[keep]
                scores_keep = scores_keep[keep]
                box_indices_keep = box_indices_keep[keep]

            boxes_results.append(torch.cat([box_indices_keep[..., None], boxes_keep], dim = -1))
            scores_results.append(scores_keep)
            indices_results.append(box_indices_keep)
        
        return boxes_results, scores_results

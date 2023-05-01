'''
3D Region Proposal Network 
Reference:
https://zhuanlan.zhihu.com/p/145842317
https://github.com/yxgeee/pytorch-FPN/blob/master/lib/nets/resnet_v1.py
https://github.com/pytorch/vision/tree/87cde716b7f108f3db7b86047596ebfad1b88380/torchvision/models/detection

'''
from torch import device
import torch.nn.functional as F
from .anchor import *
from .utils import Matcher, BalancedPositiveNegativeSampler, print_shape
from .coder import AABBCoder, MidpointOffsetCoder
from .coder.misc import obb2hbb_3d, obb2points_3d
from .rotated_iou.oriented_iou_loss import cal_iou_3d, cal_giou_3d, cal_diou_3d

from .utils import batched_nms, clip_boxes_to_mesh, remove_small_boxes, box_iou_3d, batched_box_iou


def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, W: int, H: int, D: int) -> Tensor:
    '''
    Permute and flatten from (N, A*C, W, H, D) to (N, W*H*D*A, C)
    '''
    layer = layer.view(N, -1, C, W, H, D)
    layer = layer.permute(0, 3, 4, 5, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def normalize(x, axis=-1, order=2):
    l2 = np.linalg.norm(x, order, axis)
    l2 = np.expand_dims(l2, axis)
    l2[l2 == 0] = 1
    return x / l2,


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


def get_w2cs(res: int = 160):
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


def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor], num_bbox_digits: int):
    """ Concatenate outputs of different scenes. box_regression_flattened will keep list format while box_regression will be concatenated as a tensor """
    box_cls_flattened = []
    box_regression_flattened = []

    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        N, AxC, W, H, D = box_cls_per_level.shape
        A_x_NumBboxDigits = box_regression_per_level.shape[1]
        A = A_x_NumBboxDigits // num_bbox_digits
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, W, H, D)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, num_bbox_digits, W, H, D)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, num_bbox_digits)
    
    return box_cls, box_regression, box_regression_flattened


class RotatedIOULoss(nn.Module):
    '''
    3D IoU loss for OBB
    '''
    def __init__(self, loss_type="iou"):
        super(RotatedIOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        if self.loss_type == 'iou' or self.loss_type == 'linear_iou':
            ious, _, _, _, unions = cal_iou_3d(pred.unsqueeze(0), target.unsqueeze(0), verbose=True)

            volume_intersect = ious * unions        
            ious = (volume_intersect + 1.0) / (unions + 1.0)

            if self.loss_type == 'iou':
                losses = -torch.log(ious)
            elif self.loss_type == 'linear_iou':
                losses = 1 - ious

        elif self.loss_type == 'giou':
            losses, _, _ = cal_giou_3d(pred.unsqueeze(0), target.unsqueeze(0))
        elif self.loss_type == 'diou':
            losses, _ = cal_diou_3d(pred.unsqueeze(0), target.unsqueeze(0))
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        anchor_generator: AnchorGenerator3D,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_mesh: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
        iou_batch_size: int = 16,
        rotated_bbox: bool = False,
        reg_loss_type: str = "smooth_l1",
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head

        # gt is rotated or not
        self.rotate = rotated_bbox
        self.box_coder = AABBCoder() if not rotated_bbox else MidpointOffsetCoder()
        self.num_bbox_digits = 6 if not rotated_bbox else 7
        self.num_delta_digits = 6 if not rotated_bbox else 8

        # used during training
        self.box_similarity = batched_box_iou
        self.iou_batch_size = iou_batch_size

        self.proposal_matcher = Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_mesh, positive_fraction)
        
        # The parameters below are used for evaluation
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        
        self.reg_loss_type = reg_loss_type
        self.rotated_iou_loss = None
        if rotated_bbox and reg_loss_type != "smooth_l1":
            self.rotated_iou_loss = RotatedIOULoss(loss_type=reg_loss_type)
        
        self.min_size = 1e-3

    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def get_padding_masks(self, meshes, features, ori_sizes):
        masks = self.anchor_generator.get_padding_masks(meshes, features, ori_sizes)
        masks_flattened = []
        for masks_per_level in masks:
            N, A, W, H, D = masks_per_level.shape
            masks_flattened.append(permute_and_flatten(masks_per_level, N, A, 1, W, H, D))

        return torch.cat(masks_flattened, dim=1).squeeze(-1)

    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Tensor], padding_masks: Optional[List[Tensor]] = None
    ) -> Tuple[List[Tensor], List[Tensor]]:

        labels = []
        matched_gt_boxes = []
        for i, (anchors_per_mesh, gt_boxes) in enumerate(zip(anchors, targets)):
            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_mesh.device
                matched_gt_boxes_per_mesh = torch.zeros(anchors_per_mesh.shape, dtype=torch.float32, device=device)
                labels_per_mesh = torch.zeros((anchors_per_mesh.shape[0],), dtype=torch.float32, device=device)
            else:
                if gt_boxes.size(1) == 7:
                    # use rectified gt boxes to compute similarity with anchors
                    gt_boxes_aabb = obb2hbb_3d(gt_boxes)
                    match_quality_matrix = self.box_similarity(gt_boxes_aabb, anchors_per_mesh, self.iou_batch_size)
                else:
                    match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_mesh, self.iou_batch_size)
                
                if padding_masks is not None:
                    # Ensure that masked anchors are ignored in the matcher
                    mask = ~padding_masks[i]
                    match_quality_matrix[:, mask] = -1.0

                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_mesh = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_mesh = matched_idxs >= 0
                labels_per_mesh = labels_per_mesh.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_mesh[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_mesh[inds_to_discard] = -1.0

            if padding_masks is not None:
                mask = ~padding_masks[i]
                labels_per_mesh[mask] = -1.0

            labels.append(labels_per_mesh)
            matched_gt_boxes.append(matched_gt_boxes_per_mesh)

        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), ob.size(1))
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        level_indexes: Tensor,
        mesh_shapes: List[Tuple[int, int, int]],
        num_anchors_per_level: List[int],
        padding_masks: Optional[Tensor] = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        num_meshes = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_meshes, -1)
        
        level_indexes = level_indexes.detach()

        # Mask out anchors in the padded voxels
        if padding_masks is not None:
            objectness[~padding_masks] = -torch.inf

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        mesh_range = torch.arange(num_meshes, device=device)
        batch_idx = mesh_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]

        proposals = proposals[batch_idx, top_n_idx]
        level_indexes = level_indexes[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        final_indexes = []
        for boxes, scores, indexes, lvl, mesh_shape in zip(proposals, objectness_prob, level_indexes, levels, mesh_shapes):
            boxes = clip_boxes_to_mesh(boxes, mesh_shape)
            # remove small boxes
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, scores, indexes, lvl = boxes[keep], scores[keep], indexes[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            
            boxes, scores, indexes, lvl = boxes[keep], scores[keep], indexes[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores, indexes = boxes[keep], scores[keep], indexes[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
            final_indexes.append(indexes)

        return final_boxes, final_scores, final_indexes

    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor], 
        pred_bbox: Tensor, target_bbox: List[Tensor], max_mesh_dim: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])
            max_mesh_dim (int): maximum dimension of the mesh

        Returns:
            objectness_loss (Tensor)
            box_reg_loss_3d (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        if self.reg_loss_type == "smooth_l1":
            box_reg_loss_3d = (
                F.smooth_l1_loss(
                    pred_bbox_deltas[sampled_pos_inds],
                    regression_targets[sampled_pos_inds],
                    beta=1 / 9,
                    reduction="sum",
                )
                / (sampled_inds.numel())
            )
        else:
            target_bbox_cat = torch.cat(target_bbox, dim=0)
            box_reg_loss_3d = (
                self.rotated_iou_loss(
                    pred_bbox[sampled_pos_inds], 
                    target_bbox_cat[sampled_pos_inds]
                ) / (sampled_inds.numel())
            )

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        # 2d reg loss
        w, h, fx, fy = 640, 480, 600, 600
        K = torch.tensor([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=torch.float32)
        if torch.cuda.is_available():
            K = K.cuda()
        pose_list = get_w2cs(res=max_mesh_dim)
        
        target_bbox = torch.cat(target_bbox, dim=0)

        if target_bbox.size(1) == 6:
            # [xmin, ymin, zmin, xmax, ymax, zmax]
            pred_bbox = torch.cat([pred_bbox[sampled_pos_inds, :3], pred_bbox[sampled_pos_inds,3:]], dim=0)
            target_bbox = torch.cat([target_bbox[sampled_pos_inds, :3], target_bbox[sampled_pos_inds,3:]], dim=0)
        elif target_bbox.size(1) == 7:
            # [x, y, z, w, h, d, theta]
            pred_bbox = obb2points_3d(pred_bbox[sampled_pos_inds])
            target_bbox = obb2points_3d(target_bbox[sampled_pos_inds])

        if torch.cuda.is_available():
            pred_bbox = torch.cat([pred_bbox, torch.ones(pred_bbox.shape[0], 1).cuda()], dim=1)
            target_bbox = torch.cat([target_bbox, torch.ones(target_bbox.shape[0], 1).cuda()], dim=1)
        else:
            pred_bbox = torch.cat([pred_bbox, torch.ones(pred_bbox.shape[0], 1)], dim=1)
            target_bbox = torch.cat([target_bbox, torch.ones(target_bbox.shape[0], 1)], dim=1)
            
        pred_bbox_2d, target_bbox_2d = [], []
        for pose in pose_list:
            pred_bbox_2d.append(project(K, pose, pred_bbox))
            target_bbox_2d.append(project(K, pose, target_bbox))
        pred_bbox_2d = torch.cat(pred_bbox_2d, dim=0)
        target_bbox_2d = torch.cat(target_bbox_2d, dim=0)
        box_reg_loss_2d = F.smooth_l1_loss(pred_bbox_2d, target_bbox_2d, beta=1 / 9, reduction="sum") / \
                              sampled_pos_inds.numel() / max_mesh_dim
        

        return objectness_loss, box_reg_loss_3d, box_reg_loss_2d

    def forward(
        self,
        meshes: Tensor,
        features: List[Tensor],
        original_mesh_sizes: List[Tuple[int, int, int]],
        targets: Optional[List[Tensor]] = None,
        objectness_output_paths=None, 
    ) -> Tuple[List[Tensor], Dict[str, Tensor], List[Tensor]]:

        """
        Args:
            meshes (Tensor): grids for which we want to compute the predictions
            features (List[Tensor]): features computed from the grids that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            original_mesh_sizes (List[Tuple[int, int, int]]): original mesh sizes
                before padding
            targets (List[Tensor])(optional): ground-truth boxes present in the scene .
            objectness_output_paths (str)(optional): See rpn_network/run_rpn.py and rpn_network/nerf_rpn.py for more details. 

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per mesh.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
            scores (List[Tensor]): the scores for each predicted box.
        """
        # RPN uses all feature maps that are available
        objectness, pred_bbox_deltas = self.head(features)
        anchors, non_cat_anchors = self.anchor_generator(meshes, features)

        if objectness_output_paths is not None:
            self.output_objectness(objectness, original_mesh_sizes, objectness_output_paths)
        
        num_meshes = len(anchors)
        mesh_sizes = [meshes[i].shape[1:] for i in range(num_meshes)]
        max_mesh_dim = max([max(mesh_size) for mesh_size in mesh_sizes])
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] * s[3] for s in num_anchors_per_level_shape_tensors]

        objectness, pred_bbox_deltas, pred_bbox_deltas_unflattened = concat_box_prediction_layers(
            objectness, pred_bbox_deltas, self.num_delta_digits
        )
        
        padding_masks = self.get_padding_masks(meshes, features, original_mesh_sizes) if num_meshes > 1 else None

        boxes, scores, level_indexes = None, None, None
        losses = {}
        
        # During training, boxes pred and scores are unused.
        if not self.training:
            proposals = self.box_coder.decode_list(pred_bbox_deltas_unflattened, non_cat_anchors)
            level_indexes = proposals[..., -1]
            proposals = proposals[..., :self.num_bbox_digits]
            boxes, scores, level_indexes = self.filter_proposals(proposals, objectness, level_indexes, mesh_sizes, 
                num_anchors_per_level, padding_masks)

        if self.training:
            if targets is None:
                raise ValueError("targets should not be None")
            
            # 2D projection loss will backprop through the proposals
            proposals = self.box_coder.decode(pred_bbox_deltas, anchors)
            proposals = proposals.view(-1, self.num_bbox_digits)

            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets, padding_masks)
            
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

            loss_objectness, loss_rpn_box_reg, loss_rpn_box_reg_2d = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets, proposals, 
                matched_gt_boxes, max_mesh_dim
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_rpn_box_reg_2d": loss_rpn_box_reg_2d,
            }
        
        return boxes, level_indexes, losses, scores
    
    def output_objectness(self, objectness, ori_sizes, output_paths):
        for i in range(len(ori_sizes)):
            all_levels = {}
            for level in range(len(objectness)):
                score = objectness[level][i].max(dim=0)[0]
                w, l, h = np.ceil(np.array(ori_sizes[i])/2**(level+2)).astype(int)
                score = score[:w, :l, :h]
                score = score.cpu().numpy()
                all_levels[str(level)] = score
            
            output_path = output_paths[i]
            np.savez_compressed(output_path, **all_levels)

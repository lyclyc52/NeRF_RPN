import torch
import numpy as np
from torch import device, nn, normal
import torch.nn.functional as F
from .utils import *
from .coder.AABB_coder import AABBCoder
from .coder.rotated_coder import RotatedCoder
from .feature_extractor import *
from .level_mapper import _setup_scales

    
class ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.

    Args:
        nclasses: the total num of class
        batch_size: the batch size of the classification network
    """

    def __init__(self, nclasses, batch_size=1000, fg_fraction = 0.5, fg_threshold = 0.5, bg_threshold = 0.2, is_rotated_bbox = False):
        super(ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.batch_size = batch_size
        self.fg_fraction = fg_fraction
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.is_rotated_bbox = is_rotated_bbox
        
        self.bbox_size = 7 if is_rotated_bbox else 6

    """
    Args:
        all_rois(List[List[Tensor]]):  region of interest proposed by rpn network
        gt_boxes(List[Tensor[N, 7 or 6]]): the ground truth bounding boxes of each images
        gt_labels(List[Tensor[N]]): the label of each ground truch bounding boxes
    Return:
        labels:the ground truth label of each roi
        rois: rois sampled from all rois
        gt_rois: ground truth bounding boxes 
    """

    def forward(self, all_rois, gt_boxes, gt_labels, is_sample):

        assert len(all_rois) == len(gt_boxes) and len(gt_boxes) == len(gt_labels)        
        num_mesh = len(all_rois)
        
        rois_per_image = int(self.batch_size / num_mesh)
        fg_rois_per_image = int(np.round(self.fg_fraction * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        
        labels, rois, gt_rois = self._sample_rois_pytorch(
            all_rois, gt_boxes, gt_labels, fg_rois_per_image,
            rois_per_image, self._num_classes, is_sample)
        # bbox_outside_weights = (bbox_inside_weights > 0).float()

        return labels, rois, gt_rois

    def _sample_rois_pytorch(self, all_rois, gt_boxes, gt_label, fg_rois_per_image, rois_per_image, num_classes, is_sample):
        """
        Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        batch_size = len(all_rois)
        overlaps, max_overlaps, gt_assignment, labels = [], [], [], []

        cur_device = all_rois[0][0].get_device()
        for i in range(batch_size):
            overlaps.append(batched_box_iou(all_rois[i][..., 1:], gt_boxes[i]).to(cur_device))
            cur_max_overlaps, cur_gt_assignment = torch.max(overlaps[i], 1)
            max_overlaps.append(cur_max_overlaps)
            gt_assignment.append(cur_gt_assignment)
            
            cur_gt_label = gt_label[i].repeat([cur_max_overlaps.size(0),1]).to(cur_max_overlaps.get_device())
            cur_gt_assignment = cur_gt_assignment[:, None]

            cur_label = cur_gt_label.gather(1, cur_gt_assignment)
            labels.append(cur_label[..., 0])

        # The dimemsion of roi is 7 in the case. We need one digit to store the feature level of each 
        # roi. The first digit of gt_roi is useless. We just keep it to match the form. 
        if is_sample:
            labels_batch = labels[0].new(batch_size, rois_per_image).zero_()
            rois_batch  = all_rois[0].new(batch_size, rois_per_image, all_rois[0].shape[-1]).zero_()
            gt_rois_batch = all_rois[0].new(batch_size, rois_per_image, gt_boxes[0].shape[-1]).zero_()
            
            # Guard against the case when an image has fewer than max_fg_rois_per_image
            # foreground RoIs

            for i in range(batch_size):
                fg_inds = torch.nonzero(max_overlaps[i] >= self.fg_threshold).view(-1)
                fg_num_rois = fg_inds.numel()
                # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
                bg_inds = torch.nonzero(max_overlaps[i] < self.bg_threshold).view(-1)
                bg_num_rois = bg_inds.numel()

                if fg_num_rois > 0 and bg_num_rois > 0:
                    # sampling fg
                    fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                    rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes[0]).long()
                    fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                    # sampling bg
                    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                    # Seems torch.rand has a bug, it will generate very large number and make an error.
                    # We use numpy rand instead.
                    #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                    rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                    rand_num = torch.from_numpy(rand_num).type_as(gt_boxes[0]).long()
                    bg_inds = bg_inds[rand_num]

                elif fg_num_rois > 0 and bg_num_rois == 0:
                    # sampling fg
                    #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                    rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                    rand_num = torch.from_numpy(rand_num).type_as(gt_boxes[0]).long()
                    fg_inds = fg_inds[rand_num]
                    fg_rois_per_this_image = rois_per_image
                    bg_rois_per_this_image = 0

                elif bg_num_rois > 0 and fg_num_rois == 0:
                    # sampling bg
                    #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                    rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                    rand_num = torch.from_numpy(rand_num).type_as(gt_boxes[0]).long()

                    bg_inds = bg_inds[rand_num]
                    bg_rois_per_this_image = rois_per_image
                    fg_rois_per_this_image = 0

                else:
                    raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

                # The indices that we're selecting (both fg and bg)
                keep_inds = torch.cat([fg_inds, bg_inds], 0)
                # Select sampled values from various arrays:
                labels_batch[i].copy_(labels[i][keep_inds])

                # Clamp labels for the background RoIs to 0
                if fg_rois_per_this_image < rois_per_image:
                    labels_batch[i][fg_rois_per_this_image:] = 0

                rois_batch[i] = all_rois[i][keep_inds]
                gt_rois_batch[i,:, ] = gt_boxes[i][gt_assignment[i][keep_inds]]
        else:
            labels_batch = []
            rois_batch  = []
            gt_rois_batch = []

            for i in range(batch_size):
                fg_inds = torch.nonzero(max_overlaps[i] >= self.fg_threshold).view(-1)
                fg_num_rois = fg_inds.numel()
                # Clamp labels for the background RoIs to 0
                labels_batch.append(labels[0].new(all_rois[i].size(0)).zero_())
                labels_batch[i][fg_inds] = 1
                
                rois_batch.append(all_rois[i])

                gt_rois_batch.append(all_rois[0].new(all_rois[i].size(0), gt_boxes[0].shape[-1]).zero_())
                gt_rois_batch[i] = gt_boxes[i][gt_assignment[i]]

        return labels_batch, rois_batch, gt_rois_batch


class ROIPool(nn.Module):
    def __init__(self, output_size= [1,1,1], spatial_scale = [1,1,1,1], enlarge_scale = 0.2, is_rotated_bbox = False, 
                 feature_extracting_type = 'pooling', max_res = 200, remap = False, use_cuda = False):
        '''
        Args:
            output_size(List(int)): output size of output feature
            spatial_scale(List(int)): the ratio of input mesh and feature map. 
            enlarge_scale(float): the scale to enlarge the size of the input rois
            is_rotated_bbox(bool): the input rois is (x,y,z,w,h,d,theta) if it is True.
            feature_extracting_type(['pooling', 'interpolation']): the strategy that will be used to get the features of rois. 
        '''
        super(ROIPool, self).__init__()
        self.output_size = torch.tensor(output_size).type(torch.float)
        self.spatial_scale = spatial_scale
        self.enlarge_scale = enlarge_scale
        self.is_rotated_bbox = is_rotated_bbox
        self.feature_extracting_type = feature_extracting_type 
        self.canonical_scale = max_res
        self.canonical_level = len(spatial_scale)
        self.remap = remap
        self.use_cuda = use_cuda
        if use_cuda:
            from .rotated_align.roi_align_rotate_3d import ROIAlignRotated3D
            self.align = ROIAlignRotated3D(output_size, sampling_ratio = 0)

    def enlarge_roi(self,roi):
        '''
        It will enlarge the size of rois
        '''
        if self.is_rotated_bbox:
            base_cube = roi
            base_cube[..., 3:6] = base_cube[..., 3:6]  * (1 + self.enlarge_scale)
        else:
            extent = (roi[..., 3:] - roi[..., :3]) / 2  * (1 + self.enlarge_scale)
            offset = (roi[..., 3:] + roi [..., :3]) / 2
            
            base_cube = torch.tensor([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]).to(roi.device)
            base_cube = base_cube.repeat([roi.size(0), 1])
            base_cube[..., :3] = base_cube[..., :3] * extent + offset
            base_cube[..., 3:] = base_cube[..., 3:] * extent + offset

        return base_cube

    def forward(self, feature, rois, original_size = None):
        '''
        Args:
            feature(List[List[Tensor]]): feature map pyramid
            rois(List[Tensor]): ROIs selected by proposal target larger
            original_size(List(List))(Optional): original size of the feature maps. Only used when reamp is enabled.
        '''

        # If remap is True, it will run FPN verison of mapping.
        if self.remap:
            if original_size is None:
                for f in feature:
                    original_size.append(f.shape[1:] * self.spatial_scale[0])

            scales = [1/s for s in self.spatial_scale]
            map_levels = _setup_scales(
                scales, self.canonical_scale, self.canonical_level
            )
            rois_shape = [r.shape for r in rois]
            rois = [r.reshape([-1, r.shape[-1]]) for r in rois]
            rois = [r[..., 1:] for r in rois]
            levels = [map_levels(r) for r in rois]
            rois = [torch.cat([l[..., None], r], -1) for l,r in zip(levels, rois)]
            rois = [r.reshape(s) for r,s in zip(rois, rois_shape)]
            
        
        if self.is_rotated_bbox:
            if self.use_cuda:
                return self.rotated_forward_cuda(feature, rois)
            else:
                return self.rotated_forward(feature, rois)
        else:
            return self.normal_forward(feature, rois)

    def rotated_forward_cuda(self, features: List[List[Tensor]], rois: List[Tensor]):
        batch_features = []
        for f, r in zip(features, rois):
            roi_features = []
            for l in range(self.canonical_level):
                level_f = f[l][None,...].to(torch.float)
                level_r = r[r[..., 0].type(torch.long) == l] 
                level_r[..., 0] = 0.               
                level_r[..., 1:] = self.enlarge_roi(level_r[..., 1:])
                if(level_r.size(0) > 0):
                    level_r = level_r.to(torch.float)
                    roi_features.append(self.align(level_f, level_r, float(1 / self.spatial_scale[l])))
            batch_features.append(torch.cat(roi_features))
        
        return batch_features

    # Non-cuda version
    def rotated_forward(self, features: List[List[Tensor]], rois: List[Tensor]):

        batch_size = len(rois)
        cur_device = rois[0].get_device()
        spatial_scale = torch.tensor(self.spatial_scale).to(cur_device)
        batch_features = []
        
        func = [torch.floor, torch.ceil]
        eight_pts_func = []
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    eight_pts_func.append([func[a], func[b], func[c]])
                    

        for i in range(batch_size):
            level_indices = rois[i][..., 0].type(torch.long).to(cur_device)
            mesh_rois = self.enlarge_roi(rois[i][..., 1:])
            # print('mesh_rois', mesh_rois.shape)
            batch_indices = []
            roi_features = []
            
            # Parallel (parts of) features abstracting operations at each level 
            for level in range(len(features[i])):
                level_features = features[i][level]
                level_features_size = level_features.shape # [256, 40, 40, 40]
                level_mask = level_indices == level 
                level_rois = mesh_rois[level_mask] # (x,y,z,w,h,d,theta)

                # print('level_rois', level_rois)
                
                if level_rois.shape[0] == 0:
                    continue

                # We need to store the original positions of rois because they corresponds to the ground truth
                # bboxes and the labels.
                batch_indices.append(level_mask.nonzero())
                
                rois_grid_size = torch.ceil(level_rois[:, 3:6] / spatial_scale[level]).to(torch.long)
                rois_grid_size[rois_grid_size < 1] = 1

                # Generate grids with maximum size which can contains the features of all kinds of bounding boxes
                max_grid_size,_ = torch.max(rois_grid_size, dim=0)
                grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(max_grid_size[0]), torch.arange(max_grid_size[1]), torch.arange(max_grid_size[2]), indexing='ij')

                position = torch.stack([grid_x, grid_y, grid_z], dim = 0).to(cur_device)
                # print('position1', position.shape)
                position = position.reshape([3, -1]).to(torch.float)
                # print('position2', position.shape)
                position = position[None, ...].repeat([rois_grid_size.size(0), 1,1])
                # print('position', position.shape)
                

                offset = level_rois [..., :3]
                theta = level_rois[ ..., 6, None] # temp: negate theta
                zeros_tensor = torch.zeros([theta.size(0), 1]).to(cur_device)
                ones_tensor = torch.ones([theta.size(0), 1]).to(cur_device)
                rotation_mat = torch.stack([torch.cat([torch.cos(theta), -torch.sin(theta), zeros_tensor],dim = 1),
                                            torch.cat([torch.sin(theta), torch.cos(theta), zeros_tensor], dim = 1),
                                            torch.cat([zeros_tensor, zeros_tensor, ones_tensor], dim = 1)],      
                                            dim = -1).to(torch.float)

                rotation_mat = rotation_mat.permute(0,2,1)
                
                
                # Shift the grid to make the center of the actual bounding box overlap with the original
                position = position - (rois_grid_size[..., None] - 1)/ 2.  


                position = rotation_mat @ position
                position = position + offset[..., None] / spatial_scale[level]
                
                
                position = position.permute(1, 0, 2)
                position_shape = position.shape
                position = position.reshape(3, -1)
                # print(position.shape)
                all_grid_features = 0.
                

                position_mask = ((position[0] >= 0) & (position[0] <= level_features_size[1] - 1)) & \
                                    ((position[1] >= 0) & (position[1] <= level_features_size[2] - 1)) & \
                                    ((position[2] >= 0) & (position[2] <= level_features_size[3] - 1))

                # Using interpolation to get the feature at each position
                for a in range(len(eight_pts_func)):
                    interpolation_pts_list = [eight_pts_func[a][0](position[0]), eight_pts_func[a][1](position[1]), eight_pts_func[a][2](position[2])]
                    interpolation_pts = torch.stack([torch.clip(interpolation_pts_list[0], min = 0, max = level_features_size[1] - 1), 
                                                        torch.clip(interpolation_pts_list[1], min = 0, max = level_features_size[2] - 1), 
                                                        torch.clip(interpolation_pts_list[2], min = 0, max = level_features_size[3] - 1)]).to(torch.long)


                    interpolation_pts_feature = level_features[:, interpolation_pts[0], interpolation_pts[1], interpolation_pts[2]]
                    # interpolation_pts_feature = interpolation_pts_feature * interpolation_mask[None, ...]

                    scale = (position[0] - interpolation_pts_list[0]).abs() * \
                            (position[1] - interpolation_pts_list[1]).abs() * \
                            (position[2] - interpolation_pts_list[2]).abs()
                    
                    all_grid_features = all_grid_features + interpolation_pts_feature * (1. - scale[None, ...])
                
                all_grid_features = all_grid_features * position_mask[None, ...]
                all_grid_features = all_grid_features / len(eight_pts_func)
                

                all_grid_features = all_grid_features.reshape(all_grid_features.shape[:1]  + position_shape[1:2] + tuple(max_grid_size.tolist()))                
                all_grid_features = all_grid_features.permute(1, 0, 2, 3, 4)

                
                # Since the size of bboxes are different and the sizes of some features are even smaller than
                # the output size, I cannot parallel the operations here
                for roi_index in range(all_grid_features.size(0)):
                    grid_size = rois_grid_size[roi_index]
                    
                    # Finally, only a part of the whole grid will be used 
                    grid_feature = all_grid_features[roi_index][:, :grid_size[0], :grid_size[1], :grid_size[2]]
                    
                    if self.feature_extracting_type == 'pooling':
                        kernel_size = torch.ceil(grid_size / self.output_size.to(cur_device)).type(torch.int)
                        padding = (kernel_size * self.output_size.to(cur_device) - grid_size).type(torch.int)
                        extracting_layer = nn.MaxPool3d(kernel_size = kernel_size.tolist(), stride=kernel_size.tolist())
                        
                        grid_feature = torch.nn.functional.pad(grid_feature, (0,int(padding[2]), 0,int(padding[1]), 0,int(padding[0])))
                        extracted_feat = extracting_layer(grid_feature[None,...])
                    elif self.feature_extracting_type == 'interpolation':
                        extracted_feat = nn.functional.interpolate(grid_feature[None, ...], size = tuple(self.output_size.to(torch.long).tolist()), mode = 'trilinear', align_corners = True)
                    else:
                        raise NameError('Unkown feature_extracting_type') 
                    
                    roi_features.append(extracted_feat)

            roi_features = torch.cat(roi_features)
            batch_indices = torch.cat(batch_indices)
            
            # Rearrage
            arrange = torch.zeros_like(roi_features)
            arrange[batch_indices.flatten()] = roi_features
            batch_features.append(arrange)
            
       
        return batch_features
    
    # Non-cuda version
    def normal_forward(self, features, rois):
        batch_size = len(rois)
        spatial_scale = torch.tensor(self.spatial_scale).to(rois[0].get_device())
        batch_features = []
        for i in range(batch_size):
            level_index = rois[i][..., 0].type(torch.long).to(rois[0].get_device())
            level_index = level_index[:, None]
            cur_spatial_scale = spatial_scale.repeat([rois[i].size(0), 1]).gather(1, level_index).to(rois[0].get_device())
            position = self.enlarge_roi(rois[i][..., 1:])
            position = position / cur_spatial_scale
            position[..., :3] = torch.floor(position[..., :3])
            position[..., 3:] = torch.floor(position[..., 3:])
            position = position.to(torch.long)
            roi_features = []
            for j in range(position.size(0)):

                cur_feat = features[i][level_index[j]][..., 
                                                    position[j][0]: position[j][3] + 1,
                                                    position[j][1]: position[j][4] + 1,
                                                    position[j][2]: position[j][5] + 1]
                cur_feat_size = torch.tensor(cur_feat.shape[1:])
                kernel_size = torch.ceil(cur_feat_size / self.output_size).type(torch.int)
                padding = (kernel_size * self.output_size - cur_feat_size).type(torch.int)
                pooling_layer = nn.MaxPool3d(kernel_size = kernel_size.tolist(), stride=kernel_size.tolist())

                cur_feat = torch.nn.functional.pad(cur_feat, (0,int(padding[2]), 0,int(padding[1]), 0,int(padding[0])))
                pooling_feat = pooling_layer(cur_feat)
                roi_features.append(pooling_feat)

            batch_features.append(torch.stack(roi_features))

        return batch_features
    

class RCNN(nn.Module):
    def __init__(self, input_dim, block, n_classes, input_size, is_add_layer = False, is_rotated_bbox = False, is_flatten = True):
        super(RCNN, self).__init__()

        self.in_planes = input_dim
        self.n_classes = n_classes
        self.layer = None
        self.is_rotated_bbox = is_rotated_bbox
        self.is_flatten = is_flatten
        self.reg_dim = 7 if is_rotated_bbox else 6
        if is_add_layer:
            
            convs = []
            for _ in range(2):
                convs.append(nn.Conv3d(self.in_planes, self.in_planes, kernel_size=3, padding=1))
                convs.append(nn.ReLU(inplace=True))

            self.layer = nn.Sequential(*convs)
            
            # self.layer = self._make_layer(block, input_dim, 1, stride=1)

        if is_flatten:
            # self.flatten_size = (input_size[0]-1) * (input_size[1] - 1) * (input_size[2] - 1)
            self.flatten_size = input_size[0] * input_size[1] * input_size[2]
            self.RCNN_bbox_pred = nn.Linear(self.in_planes * self.flatten_size, self.reg_dim)
            self.RCNN_cls_score = nn.Linear(self.in_planes * self.flatten_size, self.n_classes)
        else:
            self.RCNN_bbox_pred = nn.Linear(self.in_planes, self.reg_dim)
            self.RCNN_cls_score = nn.Linear(self.in_planes, self.n_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    
    
    def forward(self, pooling_feature):
        x = self.layer(pooling_feature) if self.layer is not None else pooling_feature
        if self.is_flatten:
            x = x.view(x.size(0), -1)
        else:
            x = x.mean(-1).mean(-1).mean(-1)
        

        pred_bbox_delta = self.RCNN_bbox_pred(x)
        pred_scores = self.RCNN_cls_score(x)
        return pred_bbox_delta, pred_scores


class Classification_Model(nn.Module):
    def __init__(self, feature_extractor, sample_model, pooling_model, RCNN_model, 
                 n_classes = 2, is_training= True, batch_size= 20, is_rotated_bbox = False) -> None:
        super(Classification_Model, self).__init__()
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.sample_model = sample_model
        self.pooling_model = pooling_model
        self.RCNN_model = RCNN_model
        
        self.num_class = n_classes
        self.score_thresh = 0.7
        # self.spatial_scale = spatial_scale
        self.is_training = is_training
        self.is_rotated_bbox = is_rotated_bbox
        self.reg_dim = 7 if is_rotated_bbox else 6
        if is_rotated_bbox:
            self.box_coder = RotatedCoder()
        else:
            self.box_coder = AABBCoder()
    def transform(self, meshes):
        """
        Do 0-padding to support different sizes of input meshes.

        Args:
            meshes (list[Tensor]): list of meshes to be transformed.
        """
        if len(meshes) > 1:
            shapes = [mesh.shape for mesh in meshes]
            target_shape = np.max(shapes, axis=0)
            # print(f'Padding to {target_shape}')
            for i, mesh in enumerate(meshes):
                meshes[i] = F.pad(mesh, (0, target_shape[-1] - mesh.shape[-1], 0, target_shape[-2] - mesh.shape[-2], 
                                  0, target_shape[-3] - mesh.shape[-3]), mode="constant", value=0)

        return meshes
        
    def compute_loss(self, pred_scores, pred_bbox_deltas, gt_labels, regression_targets):

        gt_labels = gt_labels.reshape(-1).type(torch.long)
        objectness_loss = F.cross_entropy(pred_scores, gt_labels)
        
        inds = torch.nonzero(gt_labels > 0).view(-1)
        regression_targets = regression_targets.view(-1, self.reg_dim)
        


        box_loss = (
            F.smooth_l1_loss(
                pred_bbox_deltas[inds],
                regression_targets[inds],
                beta=1 / 9,
                reduction="sum",
            )
            / (inds.numel())
        ) if inds.numel()!=0 else torch.tensor(0., dtype=regression_targets.dtype).cuda()
        
        return {
            "loss_objectness": objectness_loss,
            "loss_rpn_box_reg": box_loss,
        }
        
    def forward(self, rois, gt_bboxes, gt_bbox_labels, features, is_sample = True, is_reg = False):
        # If fine-tune backbone, the input features are raw NeRF radiance and density
        original_size = []
        if self.feature_extractor is not None:
            features_flat = [f[0] for f in features] # flatten the list
            features_flat = self.transform(features_flat)
            for f in features_flat:
                original_size.append(f.shape[1:])
            features_flat = torch.stack(features_flat, dim=0)
            features_flat = list(self.feature_extractor(features_flat))
            
            # Reorganize the features
            features = [torch.split(f, 1, dim=0) for f in features_flat]
            features = [[f.squeeze(0) for f in f_list] for f_list in features]
            features = [[f[i].to(torch.float) for f in features] for i in range(len(features[0]))]
        

        gt_labels, sample_rois, gt_bbox = self.sample_model(rois, gt_bboxes, gt_bbox_labels, is_sample = is_sample)
        pooling_features = self.pooling_model(features, sample_rois, original_size)
        pooling_features = [i.to(torch.float) for i in pooling_features ]
        
        if is_sample:
            input_batch = gt_bbox.size(0)
            sample_batch = sample_rois.size(1)
        else:
            input_batch = len(gt_bbox)    
            
        roi_batch_split = [0]
        for i in range(len(pooling_features)):
            num = pooling_features[i].size(0)
            roi_batch_split.append(roi_batch_split[i] + num)

        pooling_features = torch.cat(pooling_features, dim = 0)
        pred_bbox_deltas, pred_scores = self.RCNN_model(pooling_features)  
        
        cls_prob = F.softmax(pred_scores, 1)
        
        pred_bbox_deltas_batch = []
        sample_rois_batch = []
        pred_cls_prob_batch = []
        
        
        for i in range(input_batch):
            temp = pred_bbox_deltas[roi_batch_split[i] :  roi_batch_split[i + 1]]
            pred_bbox_deltas_batch.append(temp)
            pred_cls_prob_batch.append(cls_prob[roi_batch_split[i] : roi_batch_split[i + 1]])
            sample_rois_batch.append(sample_rois[i][..., 1:])

        proposals = []
        for i in range(len(pred_bbox_deltas_batch)):
            proposals.append(self.box_coder.decode_single(pred_bbox_deltas_batch[i], sample_rois_batch[i]))

        # compute the loss
        loss = 0.
        if is_sample:
            regression_targets = self.box_coder.encode_single(gt_bbox.reshape(-1, gt_bbox.size(-1)), sample_rois[..., 1:].reshape(-1, sample_rois.size(-1)-1))
            regression_targets = regression_targets.reshape(input_batch,sample_batch, regression_targets.size(-1))
            
            loss = self.compute_loss(
                pred_scores, pred_bbox_deltas, gt_labels, regression_targets
            )

        if is_reg:
            return [proposals, gt_labels], pred_cls_prob_batch, loss
        else:
            return [sample_rois_batch, gt_labels], pred_cls_prob_batch, loss


if __name__ == '__main__':
    sample_model = ProposalTargetLayer(2, batch_size= 1)
    pooling_model = ROIPool([5,5,5], [2,4,8])
    RCNN_model = RCNN(512, Bottleneck, 2)
    model = Classification_Model(
            sample_model,
            pooling_model,
            RCNN_model)
    roi = [[torch.cat([torch.rand([20, 3]) * 50, torch.rand([20, 3])*127 + 128], dim = 1) for _ in range(3)]]
    gt_box = [torch.cat([torch.rand([5, 3]) * 50, torch.rand([5, 3])*127 + 128], dim = 1)]
    gt_label = [torch.ones([5])]
    feature = [[torch.rand(512, 128, 128, 128), torch.rand(512, 64, 64, 64), torch.rand(512, 32, 32, 32)]]

    model(roi, gt_box, gt_label, feature)

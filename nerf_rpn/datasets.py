from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn, Tensor, normal
from tqdm import tqdm
from abc import ABC


class BaseDataset(torch.utils.data.Dataset, ABC):
    """Base dataset class"""
    def __init__(
        self, dataset_type: str = None,
        features_path: str = None, boxes_path: str = None, 
        scene_list: Optional[List[str]] = None,
        normalize_density: bool = True,
        flip_prob: float = 0.0,
        rotate_prob: float = 0.0,
        rot_scale_prob: float = 0.0,
        z_up: bool = True
    ) -> None:
        super().__init__()
        self.dataset_type = dataset_type # hypersim, 3dfront, general
        self.features_path = features_path
        self.boxes_path = boxes_path
        self.scene_list = scene_list
        self.normalize_density = normalize_density
        self.flip_prob = flip_prob  # the probability of flipping the data along the horizontal axes
        self.rotate_prob = rotate_prob  # the probability of rotating the data by 90 degrees
        self.rot_scale_prob = rot_scale_prob  # the probability of extra rotation and scaling
        self.z_up = z_up    # whether the boxes are z-up

        self.scene_data = []

    def load_single_scene(self, scene: str):
        """
        Load a single scene
        """
        if self.boxes_path is None:
            boxes = None
        else:
            boxes = torch.from_numpy(np.load(os.path.join(self.boxes_path, scene + '.npy')))

        scene_features_path = os.path.join(self.features_path, scene + '.npz')
        with np.load(scene_features_path) as features:
            rgbsigma = features['rgbsigma']
            if self.normalize_density:
                alpha = self.density_to_alpha(rgbsigma[..., -1])
                rgbsigma[..., -1] = alpha

            # From (W, L, H, C) to (C, W, L, H)
            rgbsigma = np.transpose(rgbsigma, (3, 0, 1, 2))
            rgbsigma = torch.from_numpy(rgbsigma)

            if rgbsigma.dtype == torch.uint8:
                # normalize rgbsigma to [0, 1]
                rgbsigma = rgbsigma.float() / 255.0

        return scene, rgbsigma, boxes
    
    def load_scene_data(self, preload: bool = False):
        '''
        Check scene data and load them if needed
        '''
        if self.scene_list is None:
            # if scene_list is not provided, use all scenes in feature path
            feature_names = os.listdir(self.features_path)
            self.scene_list = [f.split('.')[0] for f in feature_names if f.endswith('.npz')]

        scenes_kept = []
        for scene in tqdm(self.scene_list):
            scene_features_path = os.path.join(self.features_path, scene + '.npz')
            if not os.path.isfile(scene_features_path):
                print(f'{scene} does not have a feature file')
                continue

            if self.boxes_path is not None:
                boxes = np.load(os.path.join(self.boxes_path, scene + '.npy'))
                if boxes.shape[0] == 0:
                    print(f'{scene} does not have any boxes')
                    continue

            scenes_kept.append(scene)

        self.scene_list = scenes_kept
        if preload:
            self.scene_data = [self.load_single_scene(scene) for scene in self.scene_list]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        if self.scene_data:
            scene, rgbsigma, boxes = self.scene_data[index]
        else:
            scene = self.scene_list[index]
            _, rgbsigma, boxes = self.load_single_scene(scene)

        if self.flip_prob > 0 or self.rotate_prob > 0 or self.rot_scale_prob > 0:
            rgbsigma, boxes = self.augment_rpn_inputs(rgbsigma, boxes, self.flip_prob, 
                                                      self.rotate_prob, self.rot_scale_prob, self.z_up)
        
        return rgbsigma, boxes, scene

    def __len__(self) -> int:
        return len(self.scene_list)
    
    @staticmethod
    def augment_rpn_inputs(rgbsigma: Tensor, boxes: Tensor, flip_prob: float, 
                           rotate_prob: float, rot_scale_prob: float, 
                           z_up: bool = True) -> Tuple[Tensor, Tensor]:
        if flip_prob < 0 or flip_prob > 1:
            raise ValueError('flip_prob must be between 0 and 1, but got {}'.format(flip_prob))
        if rotate_prob < 0 or rotate_prob > 1:
            raise ValueError('rotate_prob must be between 0 and 1, but got {}'.format(rotate_prob))
        if rot_scale_prob < 0 or rot_scale_prob > 1:
            raise ValueError(f'rotate_and_scale_prob must be between 0 and 1, but got {rot_scale_prob}')
        if boxes is not None:
            assert (z_up and boxes.shape[1] == 7) or boxes.shape[1] == 6, \
                'z_up must be True when boxes are in (x, y, z, w, l, h, t) format'

        if random.random() < rotate_prob:
            if z_up:
                rgbsigma = torch.transpose(rgbsigma, 1, 2)
                rgbsigma = torch.flip(rgbsigma, [1])
            else:
                rgbsigma = torch.transpose(rgbsigma, 1, 3)
                rgbsigma = torch.flip(rgbsigma, [3])

            if boxes is not None:
                boxes = boxes.clone()
                if boxes.shape[1] == 6:
                    if z_up:
                        boxes[:, [0, 1, 3, 4]] = boxes[:, [1, 0, 4, 3]]
                        boxes[:, [0, 3]] = rgbsigma.shape[1] - boxes[:, [3, 0]]
                    else:
                        boxes[:, [0, 2, 3, 5]] = boxes[:, [2, 0, 5, 3]]
                        boxes[:, [2, 5]] = rgbsigma.shape[3] - boxes[:, [5, 2]]
                elif boxes.shape[1] == 7:
                    boxes[:, [0, 1, 3, 4]] = boxes[:, [1, 0, 4, 3]]
                    boxes[:, 0] = rgbsigma.shape[1] - boxes[:, 0]

        axes = [0, 1] if z_up else [0, 2]
        for axis in axes:
            if random.random() < flip_prob:
                rgbsigma = rgbsigma.flip(dims=[axis+1])
                if boxes is not None:
                    boxes = boxes.clone()
                    if boxes.shape[1] == 6:
                        # [xmin, ymin, zmin, xmax, ymax, zmax]
                        boxes[:, [axis, axis+3]] = rgbsigma.shape[axis+1] - boxes[:, [axis+3, axis]]
                    elif boxes.shape[1] == 7:
                        # [x, y, z, w, l, h, theta]
                        boxes[:, axis] = rgbsigma.shape[axis+1] - boxes[:, axis]
                        boxes[:, -1] = -boxes[:, -1]

        if boxes is not None and boxes.shape[1] == 7 and random.random() < rot_scale_prob:
            angle = random.uniform(-np.pi / 18, np.pi / 18)
            scale = random.uniform(0.9, 1.1)
            rgbsigma, boxes = rotate_and_scale_scene(rgbsigma, boxes, angle, scale)
        
        return rgbsigma, boxes

    @staticmethod
    def density_to_alpha(density):
        return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)

    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, Tensor, Tensor]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # The network expects the features and boxes of different scenes to be in two lists
        rgbsigma = []
        boxes = []
        scenes = []
        for sample in batch:
            rgbsigma.append(sample[0])
            boxes.append(sample[1])
            scenes.append(sample[2])
        return rgbsigma, boxes, scenes


class Front3DRPNDataset(BaseDataset):
    def __init__(self, features_path: str, boxes_path: str, 
                 scene_list: Optional[List[str]] = None,
                 normalize_density: bool = True,
                 flip_prob: float = 0.0,
                 rotate_prob: float = 0.0,
                 rot_scale_prob: float = 0.0,
                 preload: bool = False):
        super().__init__('3dfront', features_path, boxes_path, scene_list, normalize_density, 
                         flip_prob, rotate_prob, rot_scale_prob)
        self.load_scene_data(preload=preload)


class HypersimRPNDataset(BaseDataset):
    def __init__(self, features_path: str, boxes_path: str, 
                 scene_list: Optional[List[str]] = None,
                 normalize_density: bool = True,
                 flip_prob: float = 0.0,
                 rotate_prob: float = 0.0,
                 rot_scale_prob: float = 0.0,
                 preload: bool = False):
        super().__init__('hypersim', features_path, boxes_path, scene_list, 
                         normalize_density, flip_prob, rotate_prob, rot_scale_prob)
        self.load_scene_data(preload=preload)


class ScanNetRPNDataset(BaseDataset):
    def __init__(self, scene_list: Optional[List[str]],
                 features_path: str, boxes_path: str, 
                 flip_prob: float = 0.0,
                 rotate_prob: float = 0.0,
                 rot_scale_prob: float = 0.0):
        # ScanNet NeRF features are z-up already
        super().__init__('hypersim', features_path, boxes_path, scene_list, 
                         False, flip_prob, rotate_prob, rot_scale_prob, z_up=True)

        self.load_scene_data(preload=True)

        for scene in self.scene_data:
            rgbsigma = scene[1]
            density = rgbsigma.reshape(rgbsigma.shape[0], -1)[-1, :]
            alpha = self.density_to_alpha(density)
            alpha = alpha.reshape(rgbsigma.shape[1:])
            scene[1][-1, ...] = alpha

    @staticmethod
    def density_to_alpha(density):
        # ScanNet uses dense depth priors NeRF, which uses ReLU activation
        activation = np.clip(density, a_min=0, a_max=None)
        return np.clip(1.0 - np.exp(-activation / 100.0), 0.0, 1.0)


class GeneralRPNDataset(BaseDataset):
    def __init__(self, csv_path, normalize_density: bool = True) -> None:
        super().__init__('general')

        self.df = pd.read_csv(csv_path, dtype=str)
        self.normalize_density = normalize_density
        self.scene_list = []
        for row in tqdm(self.df.itertuples()):
            scene = row.scene
            rgbsigma_path = row.rgbsigma_path
            boxes_path = row.boxes_path

            self.scene_list.append(scene)
            assert os.path.isfile(rgbsigma_path), f'{rgbsigma_path} does not exist'
            
            boxes = None
            if boxes_path != 'None':
                assert os.path.isfile(boxes_path), f'{boxes_path} does not exist'
                boxes = torch.from_numpy(np.load(boxes_path))

            with np.load(rgbsigma_path) as features:
                rgbsigma = features['rgbsigma']
                if normalize_density:
                    alpha = self.density_to_alpha(rgbsigma[..., -1])
                    rgbsigma[..., -1] = alpha

                # From (W, L, H, C) to (C, W, L, H)
                rgbsigma = np.transpose(rgbsigma, (3, 0, 1, 2))
                rgbsigma = torch.from_numpy(rgbsigma)

                if rgbsigma.dtype == torch.uint8:
                    rgbsigma = rgbsigma.float() / 255.0

            self.scene_data.append((scene, rgbsigma, boxes))


def split_hypersim_dataset(scenes, train_ratio, val_ratio, output_path):
    assert train_ratio + val_ratio <= 1.0, 'train_ratio + val_ratio must be <= 1.0'

    scenes_shuffled = list(scenes)
    random.shuffle(scenes_shuffled)

    train_split = int(len(scenes_shuffled) * train_ratio)
    val_split = int(len(scenes_shuffled) * (train_ratio + val_ratio))

    train_scenes = scenes_shuffled[:train_split]
    val_scenes = scenes_shuffled[train_split:val_split]
    test_scenes = scenes_shuffled[val_split:]

    train_scenes = np.array(train_scenes)
    val_scenes = np.array(val_scenes)
    test_scenes = np.array(test_scenes)

    np.savez(os.path.join(output_path, 'hypersim_split.npz'), 
             train_scenes=train_scenes, val_scenes=val_scenes, test_scenes=test_scenes)


def rotate_and_scale_scene(rgbasigma, boxes, angle, scale):
    assert boxes is None or boxes.shape[1] == 7

    xform = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float) * scale

    # Do not use affine_grid because it shears the tensor
    res = rgbasigma.shape[1:]
    x = torch.linspace(-1, 1, res[0]) * res[0] / 2
    y = torch.linspace(-1, 1, res[1]) * res[1] / 2
    z = torch.linspace(-1, 1, res[2]) * res[2] / 2

    grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
    grid = grid.reshape(-1, 3)
    grid = grid @ xform.T
    grid = grid[..., [2, 1, 0]]
    grid = grid.reshape(res[0], res[1], res[2], 3)

    grid[..., 0] = grid[..., 0] / (res[2] / 2)
    grid[..., 1] = grid[..., 1] / (res[1] / 2)
    grid[..., 2] = grid[..., 2] / (res[0] / 2)
    grid = grid.unsqueeze(0)

    rgbasigma = F.grid_sample(rgbasigma.unsqueeze(0), grid, align_corners=True).squeeze(0)

    if boxes is not None:
        boxes = boxes.clone()
        boxes[:, 6] = boxes[:, 6] - angle
        boxes[:, 3:6] = boxes[:, 3:6] / scale

        center = torch.tensor(res).unsqueeze(0) / 2
        offset = boxes[:, :3] - center
        offset = offset @ (xform.to(boxes.dtype) / (scale * scale))
        boxes[:, :3] = offset + center

    return rgbasigma, boxes


class RPNClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, features_path: str, boxes_path: str, roi_path: str,
                 scene_names: Optional[List[str]] = None, fine_tune: bool = False,
                 normalize_density: bool = True,
                 flip_prob: float = 0.0,
                 rotate_prob: float = 0.0,
                 rotate_scale_prob: float = 0.0):
        self.features_path = features_path
        self.boxes_path = boxes_path
        self.fine_tune = fine_tune
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_scale_prob = rotate_scale_prob
        
        if scene_names is None:
            feature_names = os.listdir(features_path)
            scene_names = [f.split('.')[0] for f in feature_names if f.endswith('.npz')]

        self.scene_data = []
        for scene_name in tqdm(scene_names):
            if not os.path.isfile(os.path.join(boxes_path, scene_name + '.npy')) or \
                not os.path.isfile(os.path.join(roi_path, scene_name + '.npz')):
                print(f'{scene_name} does not have a training file')
                continue
            feature_file = os.path.join(features_path, scene_name + '.npz')
            
            with np.load(feature_file, allow_pickle=True) as features:
                resolution = features['resolution']

                if not fine_tune:
                    level_features = features['level_features'] 
                    output_features = []   
                    for i in range(len(level_features)):
                        level_features[i] = level_features[i].reshape(resolution[i]).astype(np.float32)
                        output_features.append(torch.from_numpy(level_features[i]))
                    
                else:
                    rgbsigma = features['rgbsigma'].astype(np.float32)
                    if normalize_density:
                        alpha = self.density_to_alpha(rgbsigma[..., -1])
                        rgbsigma[..., -1] = alpha
    
                    # From (W, L, H, C) to (C, W, L, H)
                    rgbsigma = np.transpose(rgbsigma, (3, 0, 1, 2))
                    rgbsigma = torch.from_numpy(rgbsigma)


            boxes = torch.from_numpy(np.load(os.path.join(boxes_path, scene_name + '.npy')))
            
            roi_file = os.path.join(roi_path, scene_name + '.npz')
            with np.load(roi_file, allow_pickle=True) as f_roi:
                level_indices = f_roi['level_indices']
                proposals = f_roi['proposals']

            if fine_tune:
                world_vol = resolution[0] * resolution[1] * resolution[2]
                proposals_vol = proposals[:, 3] * proposals[:, 4] * proposals[:, 5]
                ratio = proposals_vol / world_vol
                keep = ratio <= 0.5
                size_origin = proposals.shape[0]
                level_indices, proposals = level_indices[keep], proposals[keep]
                size_after = proposals.shape[0]
                # if size_origin > size_after:
                #     print(f'remove rois from {scene_name}')
                
            rois = torch.from_numpy(np.concatenate([level_indices[..., None], proposals], axis = 1))

            if not fine_tune:
                self.scene_data.append((scene_name, output_features, boxes, rois))
            else:
                self.scene_data.append((scene_name, [rgbsigma], boxes, rois))

    def __len__(self) -> int:
        return len(self.scene_data)

    @staticmethod
    def density_to_alpha(density):
        return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)

    def __getitem__(self, index: int) -> Tuple[List[Tensor], Tensor, Tensor, Tensor]:
        scene_name, level_features, boxes, rois = self.scene_data[index]

        if self.fine_tune:
            if self.flip_prob > 0 or self.rotate_prob > 0 or self.rotate_scale_prob > 0:
                level_indices = rois[..., :1]
                gt_size = boxes.size(0)
                aug_boxes = torch.cat([boxes, rois[..., 1:]])

                level_features, aug_boxes = self.augment_rpn_inputs(
                    level_features[0], aug_boxes, self.flip_prob, self.rotate_prob, self.rotate_scale_prob
                )
                boxes = aug_boxes[:gt_size]
                rois = torch.cat([level_indices, aug_boxes[gt_size:]], dim=-1)         
                level_features = [level_features] 
        return level_features, boxes, rois, scene_name

    @staticmethod
    def collate_fn(batch: List[Tuple[Tensor, Tensor, List[Tensor], Tensor]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        # The network expects the features and boxes of different scenes to be in two lists
        level_features = []
        boxes = []
        rois = []
        scenes = []
        for sample in batch:
            level_features.append(sample[0])
            boxes.append(sample[1])
            rois.append(sample[2])
            scenes.append(sample[3])
        return level_features, boxes, rois, scenes

    @staticmethod
    def augment_rpn_inputs(rgbsigma: Tensor, boxes: Tensor, flip_prob: float, 
                           rotate_prob: float, rot_scale_prob: float, 
                           z_up: bool = True) -> Tuple[Tensor, Tensor]:
        if flip_prob < 0 or flip_prob > 1:
            raise ValueError('flip_prob must be between 0 and 1, but got {}'.format(flip_prob))
        if rotate_prob < 0 or rotate_prob > 1:
            raise ValueError('rotate_prob must be between 0 and 1, but got {}'.format(rotate_prob))
        if rot_scale_prob < 0 or rot_scale_prob > 1:
            raise ValueError(f'rotate_and_scale_prob must be between 0 and 1, but got {rot_scale_prob}')
        if boxes is not None:
            assert (z_up and boxes.shape[1] == 7) or boxes.shape[1] == 6, \
                'z_up must be True when boxes are in (x, y, z, w, l, h, t) format'

        if random.random() < rotate_prob:
            if z_up:
                rgbsigma = torch.transpose(rgbsigma, 1, 2)
                rgbsigma = torch.flip(rgbsigma, [1])
            else:
                rgbsigma = torch.transpose(rgbsigma, 1, 3)
                rgbsigma = torch.flip(rgbsigma, [3])

            if boxes is not None:
                boxes = boxes.clone()
                if boxes.shape[1] == 6:
                    if z_up:
                        boxes[:, [0, 1, 3, 4]] = boxes[:, [1, 0, 4, 3]]
                        boxes[:, [0, 3]] = rgbsigma.shape[1] - boxes[:, [3, 0]]
                    else:
                        boxes[:, [0, 2, 3, 5]] = boxes[:, [2, 0, 5, 3]]
                        boxes[:, [2, 5]] = rgbsigma.shape[3] - boxes[:, [5, 2]]
                elif boxes.shape[1] == 7:
                    boxes[:, [0, 1, 3, 4]] = boxes[:, [1, 0, 4, 3]]
                    boxes[:, 0] = rgbsigma.shape[1] - boxes[:, 0]

        axes = [0, 1] if z_up else [0, 2]
        for axis in axes:
            if random.random() < flip_prob:
                rgbsigma = rgbsigma.flip(dims=[axis+1])
                if boxes is not None:
                    boxes = boxes.clone()
                    if boxes.shape[1] == 6:
                        # [xmin, ymin, zmin, xmax, ymax, zmax]
                        boxes[:, [axis, axis+3]] = rgbsigma.shape[axis+1] - boxes[:, [axis+3, axis]]
                    elif boxes.shape[1] == 7:
                        # [x, y, z, w, l, h, theta]
                        boxes[:, axis] = rgbsigma.shape[axis+1] - boxes[:, axis]
                        boxes[:, -1] = -boxes[:, -1]

        if boxes is not None and boxes.shape[1] == 7 and random.random() < rot_scale_prob:
            angle = random.uniform(-np.pi / 18, np.pi / 18)
            scale = random.uniform(0.9, 1.1)
            rgbsigma, boxes = rotate_and_scale_scene(rgbsigma, boxes, angle, scale)
        
        return rgbsigma, boxes

import json
import os
import glob
import torch
import numpy as np
import argparse
import logging
import importlib.util
from copy import deepcopy

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.fcos.fcos import FCOSOverNeRF
from model.feature_extractor import Bottleneck, ResNet_FPN_256, ResNet_FPN_64
from model.feature_extractor import VGG_FPN, SwinTransformer_FPN
from model.utils import box_iou_3d, print_shape
from datasets import HypersimRPNDataset, Front3DRPNDataset, GeneralRPNDataset, ScanNetRPNDataset
from eval import evaluate_box_proposals_ap, evaluate_box_proposals_recall

from tqdm import tqdm
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Train and eval the NeRF RPN baseline using FCOS.')

    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'benchmark'])
    parser.add_argument('--dataset', '--dataset_name', default='hypersim', 
                        choices=['hypersim', 'front3d', 'general', 'scannet'])

    parser.add_argument('--features_path', default='', help='The path to the features.')
    parser.add_argument('--boxes_path', default='', help='The path to the boxes.')
    parser.add_argument('--save_path', default='', help='The path to save the model.')
    parser.add_argument('--dataset_split', default='', help='The dataset split to use.')
    parser.add_argument('--checkpoint', default='', help='The path to the checkpoint to load.')
    parser.add_argument('--load_backbone_only', action='store_true', help='Only load the backbone weights.')
    parser.add_argument('--preload', action='store_true', help='Preload the features and boxes.')

    # General dataset csv files
    parser.add_argument('--train_csv', default='', help='The path to the train csv.')
    parser.add_argument('--val_csv', default='', help='The path to the val csv.')
    parser.add_argument('--test_csv', default='', help='The path to the test csv.')

    parser.add_argument('--backbone_type', type=str, default='swin_s', 
                        choices=['resnet', 'vgg_AF', 'vgg_EF',  'swin_t', 'swin_s', 'swin_b', 'swin_l'],)
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension for backbone.')
    
    parser.add_argument('--rotated_bbox', action='store_true', 
                        help='If true, bbox: (N, 7), [x, y, z, w, h, d, theta] \
                              If false, bbox: (N, 6), [xmin, ymin, zmin, xmax, ymax, zmax]')
    parser.add_argument('--resolution', type=int, default=160, help='The max resolution of the input features.')
    parser.add_argument('--normalize_density', action='store_true', help='Whether to normalize the density.')
    parser.add_argument('--output_proposals', action='store_true', 
                        help='Whether to output proposals during evaluation.')
    parser.add_argument('--save_level_index', action='store_true',
                        help='Whether to save level indices')
    parser.add_argument('--filter', choices=['none', 'tp', 'fp'], default='none', 
                        help='Filter the proposal output for visualization and debugging.')
    parser.add_argument('--filter_threshold', type=float, default=0.7,
                        help='The IoU threshold for the proposal filter, only used if --output_proposals is True '
                        'and --filter is not "none".')
    parser.add_argument('--output_voxel_scores', action='store_true',
                        help='Whether to output per-voxel objectness scores during evaluation, by default output '
                             'to save_path/voxel_scores dir.')

    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int, help='The batch size.')
    parser.add_argument('--num_epochs', default=100, type=int, help='The number of epochs to train.')
    parser.add_argument('--lr', default=1e-4, type=float, help='The learning rate.')
    parser.add_argument('--reg_loss_weight', default=1.0, type=float, 
                        help='The weight for balancing the regression loss.')
    parser.add_argument('--weight_decay', default=0.01, type=float, 
                        help='The weight decay coefficient of AdamW.')
    parser.add_argument('--clip_grad_norm', default=0.1, type=float, help='The gradient clipping norm.')

    parser.add_argument('--log_interval', default=20, type=int, help='The number of iterations to print the loss.')
    parser.add_argument('--log_to_file', action='store_true', help='Whether to log to a file.')
    parser.add_argument('--eval_interval', default=1, type=int, help='The number of epochs to evaluate.')
    parser.add_argument('--keep_checkpoints', default=1, type=int, help='The number of latest checkpoints to keep.')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')

    parser.add_argument('--rotate_prob', default=0.5, type=float, help='The probability of rotating the scene by 90 degrees.')
    parser.add_argument('--flip_prob', default=0.5, type=float, help='The probability of flipping the scene.')
    parser.add_argument('--rot_scale_prob', default=0.5, type=float, help='The probability of extra scaling and rotation.')
    
    # Distributed training parameters
    parser.add_argument('--gpus', default='', help='The gpus to use for distributed training. If empty, '
                        'uses the first available gpu. DDP is only enabled if this is greater than one.')

    # RPN head and loss parameters
    parser.add_argument('--num_convs', default=4, type=int, 
                        help='The number of common convolutional layers in the RPN head.')
    parser.add_argument('--norm_reg_targets', action='store_true',
                        help='Whether to normalize the regression targets for each feature map level.')
    parser.add_argument('--centerness_on_reg', action='store_true',
                        help='Whether to append the centerness layer to the regression tower.')
    parser.add_argument('--center_sampling_radius', default=1.5, type=float,
                        help='The radius for center sampling of the bounding box.')
    parser.add_argument('--iou_loss_type', choices=['iou', 'linear_iou', 'giou', 'diou', 'smooth_l1'], default='iou',
                        help='The type of IOU loss to use for the RPN.')
    parser.add_argument('--use_additional_l1_loss', action='store_true',
                        help='Whether to use additional smooth L1 loss for the OBB midpoint offsets.')
    parser.add_argument('--conv_at_start', action='store_true',
                        help='Whether to add convolutional layers at the start of the backbone.')
    parser.add_argument('--proj2d_loss_weight', default=0.0, type=float,
                        help='The weight for balancing the 2D projection loss.')

    # RPN testing parameters (FCOSPostProcessor)
    parser.add_argument('--pre_nms_top_n', default=2500, type=int, 
                        help='The number of top proposals to keep before applying NMS.')
    parser.add_argument('--fpn_post_nms_top_n', default=2500, type=int,
                        help='The number of top proposals to keep after applying NMS.')
    parser.add_argument('--nms_thresh', default=0.3, type=float,
                        help='The NMS threshold.')
    parser.add_argument('--pre_nms_thresh', default=0.0, type=float,
                        help='The score threshold.')
    parser.add_argument('--min_size', default=0.0, type=float,
                        help='The minimum size of a proposal.')
    parser.add_argument('--ap_top_n', default=None, type=int,
                        help='The number of top proposals to use for AP evaluation.')

    parser.add_argument('--output_all', action='store_true',
                        help='Output proposals of all sets in evaluation.')


    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, args, rank=0, world_size=1, device_id=None, logger=None):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device_id = device_id
        self.logger = logger if logger is not None else logging.getLogger()
        torch.multiprocessing.set_sharing_strategy('file_system')

        if args.wandb and rank == 0:
            wandb.init(project='nerf-rpn', config=deepcopy(args))

        self.logger.info('Constructing model...')

        self.build_backbone()

        self.model = FCOSOverNeRF(
            args=args,
            backbone=self.backbone,
            fpn_strides=[4, 8, 16, 32],
            world_size=self.world_size
        )

        if args.checkpoint:
            assert os.path.exists(args.checkpoint), 'The checkpoint does not exist.'
            self.logger.info(f'Loading checkpoint from {args.checkpoint}.')
            if args.load_backbone_only:
                self.logger.info('Loading backbone only.')
            checkpoint = torch.load(args.checkpoint, map_location='cpu')

            # print('Training args from checkpoint:')
            # print(checkpoint['train_args'])

            self.model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            if not args.load_backbone_only:
                self.model.fcos_module.load_state_dict(checkpoint['fcos_state_dict'])

        if torch.cuda.is_available():
            self.model.cuda()

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device_id], find_unused_parameters=True)

        if args.wandb and rank == 0:
            wandb.watch(self.model, log_freq=10)

        spec = importlib.util.find_spec('torchinfo')
        if spec is not None:
            import torchinfo
            input_rgbsigma = [torch.rand(4, 200, 200, 130)] * 2
            input_boxes = torch.rand(32, 6)
            input_boxes[:, 3:] += 1.0
            input_boxes = [input_boxes] * 2
            # torchinfo.summary(self.model, input_data=(input_rgbsigma, input_boxes))
        else:
            self.logger.info(self.model)

        self.init_datasets()
    
    def build_backbone(self):
        if self.args.backbone_type == 'resnet':
            self.backbone = ResNet_FPN_256(Bottleneck, [3, 4, 6, 3], input_dim=self.args.input_dim, is_max_pool=True)
        elif self.args.backbone_type == 'vgg_AF':
            self.backbone = VGG_FPN("AF", self.args.input_dim, True, self.args.resolution)
        elif self.args.backbone_type == 'vgg_EF':
            self.backbone = VGG_FPN("EF", self.args.input_dim, True, self.args.resolution)
        elif self.args.backbone_type.startswith('swin'):
            swin = {'swin_t': {'embed_dim':96, 'depths':[2, 2, 6, 2], 'num_heads':[3, 6, 12, 24]},
                    'swin_s': {'embed_dim':96, 'depths':[2, 2, 18, 2], 'num_heads':[3, 6, 12, 24]},
                    'swin_b': {'embed_dim':128, 'depths':[2, 2, 18, 2], 'num_heads':[3, 6, 12, 24]},
                    'swin_l': {'embed_dim':192, 'depths':[2, 2, 18, 2], 'num_heads':[6, 12, 24, 48]}}
            self.backbone = SwinTransformer_FPN(patch_size=[4, 4, 4], 
                                                embed_dim=swin[self.args.backbone_type]['embed_dim'], 
                                                depths=swin[self.args.backbone_type]['depths'],
                                                num_heads=swin[self.args.backbone_type]['num_heads'], 
                                                window_size=[4, 4, 4],
                                                stochastic_depth_prob=0,
                                                expand_dim=True,
                                                input_dim=self.args.input_dim)

    def init_datasets(self):
        if not self.args.dataset_split and self.args.dataset != 'general':
            raise ValueError('The dataset split must be specified if the dataset type is not general.')

        self.logger.info(f'Loading dataset split from {self.args.dataset_split}.')

        if self.args.dataset != 'general':
            with np.load(self.args.dataset_split) as split:
                self.train_scenes = split['train_scenes']
                self.test_scenes = split['test_scenes']
                self.val_scenes = split['val_scenes']

                if self.args.output_all:
                    self.test_scenes = np.concatenate([self.train_scenes,self.test_scenes,self.val_scenes])

        if self.args.mode == 'eval':
            if self.args.dataset == 'hypersim':
                self.test_set = HypersimRPNDataset(self.args.features_path, self.args.boxes_path,
                                                   normalize_density=self.args.normalize_density,
                                                   scene_list=self.test_scenes, preload=self.args.preload)
            elif self.args.dataset == 'front3d':
                self.test_set = Front3DRPNDataset(self.args.features_path, self.args.boxes_path,
                                                  normalize_density=self.args.normalize_density,
                                                  scene_list=self.test_scenes, preload=self.args.preload)
            elif self.args.dataset == 'scannet':
                self.test_set = ScanNetRPNDataset(self.test_scenes, self.args.features_path, self.args.boxes_path)
            else:
                self.test_set = GeneralRPNDataset(self.args.test_csv, self.args.normalize_density)

            if self.rank == 0:
                self.logger.info(f'Loaded {len(self.test_set)} scenes for evaluation')

    def save_checkpoint(self, epoch, path):
        if self.world_size == 1:
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': self.model.backbone.state_dict(),
                'fcos_state_dict': self.model.fcos_module.state_dict(),
                'train_args': self.args.__dict__,
            }, path)
        else:
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': self.model.module.backbone.state_dict(),
                'fcos_state_dict': self.model.module.fcos_module.state_dict(),
                'train_args': self.args.__dict__,
            }, path)

    def delete_old_checkpoints(self, path, keep_latest=5):
        files = glob.glob(f'{path}/epoch_*.pt')
        files.sort(key=os.path.getmtime)
        if len(files) > keep_latest:
            for file in files[:-keep_latest]:
                logging.info(f'Deleting old checkpoint {file}.')
                os.remove(file)

    def train_loop(self):
        if self.args.dataset == 'hypersim':
            self.train_set = HypersimRPNDataset(self.args.features_path, self.args.boxes_path, scene_list=self.train_scenes,
                                                normalize_density=self.args.normalize_density,
                                                flip_prob=self.args.flip_prob, rotate_prob=self.args.rotate_prob,
                                                rot_scale_prob=self.args.rot_scale_prob, preload=self.args.preload)

            if self.rank == 0:
                self.val_set = HypersimRPNDataset(self.args.features_path, self.args.boxes_path, scene_list=self.val_scenes, 
                                                  normalize_density=self.args.normalize_density, preload=self.args.preload)
        elif self.args.dataset == 'front3d':
            self.train_set = Front3DRPNDataset(self.args.features_path, self.args.boxes_path, scene_list=self.train_scenes, 
                                               normalize_density=self.args.normalize_density,
                                               flip_prob=self.args.flip_prob, rotate_prob=self.args.rotate_prob,
                                               rot_scale_prob=self.args.rot_scale_prob, preload=self.args.preload)

            if self.rank == 0:
                self.val_set = Front3DRPNDataset(self.args.features_path, self.args.boxes_path, scene_list=self.val_scenes,
                                                 normalize_density=self.args.normalize_density, preload=self.args.preload)
        elif self.args.dataset == 'scannet':
            self.train_set = ScanNetRPNDataset(self.train_scenes, self.args.features_path, self.args.boxes_path,
                                               flip_prob=self.args.flip_prob, rotate_prob=self.args.rotate_prob,
                                               rot_scale_prob=self.args.rot_scale_prob)
            if self.rank == 0:
                self.val_set = ScanNetRPNDataset(self.val_scenes, self.args.features_path, self.args.boxes_path)

        collate_fn = self.train_set.collate_fn

        if self.rank == 0:
            self.logger.info(f'Loaded {len(self.train_set)} training scenes, '
                             f'{len(self.val_set)} validation scenes')

        if self.world_size == 1:
            self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, 
                                           collate_fn=collate_fn,
                                           shuffle=True, num_workers=4, pin_memory=True)
        else:
            self.train_sampler = DistributedSampler(self.train_set)
            self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size // self.world_size,
                                           collate_fn=collate_fn,
                                           sampler=self.train_sampler, num_workers=2, pin_memory=True)

        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, 
                               weight_decay=self.args.weight_decay)

        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.args.lr,
                                    total_steps=self.args.num_epochs * len(self.train_loader))

        self.best_metric = None
        os.makedirs(self.args.save_path, exist_ok=True)
        # self.eval(self.val_set)

        for epoch in range(1, self.args.num_epochs + 1):
            if self.world_size > 1:
                self.train_sampler.set_epoch(epoch)

            self.train_epoch(epoch)
            if self.rank != 0:
                continue

            if epoch % self.args.eval_interval == 0 or epoch == self.args.num_epochs:
                recalls, APs = self.eval(self.val_set)
                metric = recalls[-1]
                if self.best_metric is None or metric > self.best_metric:
                    self.best_metric = metric
                    self.save_checkpoint(epoch, os.path.join(self.args.save_path, 'model_best.pt'))

                self.save_checkpoint(epoch, os.path.join(self.args.save_path, f'epoch_{epoch}.pt'))
                self.delete_old_checkpoints(self.args.save_path, keep_latest=self.args.keep_checkpoints)

    def train_epoch(self, epoch):
        # torch.autograd.set_detect_anomaly(True)
        for i, batch in enumerate(self.train_loader):
            # self.logger.debug(f'GPU {self.device_id} Epoch {epoch} Iter {i} {batch[-1]} '
            #                   f'Grid size: {[x.shape for x in batch[0]]}, GT boxes: {[x.shape for x in batch[1]]}')

            self.model.train()
            self.optimizer.zero_grad()

            rgbsigma, boxes, scene_name = batch
            
            if torch.cuda.is_available():
                rgbsigma = [item.cuda() for item in rgbsigma]
                boxes = [item.cuda() for item in boxes]

            proposals, losses, scores = self.model(rgbsigma, boxes)
            losses['loss_reg'] *= self.args.reg_loss_weight
            loss = losses['loss_cls'] + losses['loss_reg'] + losses['loss_centerness']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            self.logger.debug(f'GPU {self.device_id} Epoch {epoch} Iter {i} {batch[-1]} '
                              f'loss_cls: {losses["loss_cls"].item():.6f}, loss_reg: {losses["loss_reg"].item():.6f}, '
                              f'loss_centerness: {losses["loss_centerness"].item():.6f}')

            if self.world_size > 1:
                dist.barrier()
                dist.all_reduce(losses['loss_cls'])
                dist.all_reduce(losses['loss_reg'])
                dist.all_reduce(losses['loss_centerness'])
                dist.all_reduce(loss)

                losses['loss_cls'] /= self.world_size
                losses['loss_reg'] /= self.world_size
                losses['loss_centerness'] /= self.world_size
                loss /= self.world_size

            if i % self.args.log_interval == 0 and self.rank == 0:
                self.logger.info(f'epoch {epoch} [{i}/{len(self.train_loader)}]  '
                                 f'lr: {self.scheduler.get_last_lr()[0]:.6f}  '
                                 f'loss: {loss.item():.4f}')

            if self.args.wandb and self.rank == 0:
                wandb.log({
                    'lr': self.scheduler.get_last_lr()[0],
                    'loss': loss.item(),
                    'loss_cls': losses['loss_cls'].item(),
                    'loss_reg': losses['loss_reg'].item(),
                    'loss_centerness': losses['loss_centerness'].item(),
                    'epoch': epoch,
                    'iter': i,
                })

    def output_proposals(self, scenes, proposals, scores, gt_boxes):
        output_path = os.path.join(self.args.save_path, 'proposals')
        os.makedirs(output_path, exist_ok=True)
        for scene, proposal, score, gt in zip(scenes, proposals, scores, gt_boxes):
            if self.args.save_level_index:
                level_index = proposal[..., 0]
                proposal = proposal[..., 1:]
            if self.args.filter != 'none':
                if proposal.shape[0] == 0:
                    print(f'No proposals for {scene}')
                    continue
                if gt.shape[0] == 0:
                    print(f'No GT for {scene}')
                    continue

                iou = box_iou_3d(gt, proposal)
                max_iou, _ = iou.max(dim=0)
                keep = max_iou > self.args.filter_threshold
                if self.args.filter == 'fp':
                    keep = ~keep
                proposal = proposal[keep]
                level_index = level_index[keep]
                score = score[keep]
            if self.args.save_level_index:
                np.savez(os.path.join(output_path, f'{scene}.npz'), proposals=proposal, scores=score, level_indices=level_index)
            else:
                np.savez(os.path.join(output_path, f'{scene}.npz'), proposals=proposal, scores=score)

    @torch.no_grad()
    def eval(self, dataset):
        self.model.eval()
        collate_fn = dataset.collate_fn
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size // self.world_size, 
                                shuffle=False, num_workers=4,
                                collate_fn=collate_fn)

        self.logger.info(f'Evaluating...')

        proposals_list = []
        scores_list = []
        gt_boxes_list = []
        scenes_list = []

        for batch in tqdm(dataloader):
            rgbsigma, gt_boxes, scenes = batch
            if torch.cuda.is_available():
                rgbsigma = [item.cuda() for item in rgbsigma]

            output_paths = None
            if self.args.output_voxel_scores:
                scores_dir = os.path.join(self.args.save_path, 'voxel_scores')
                os.makedirs(scores_dir, exist_ok=True)
                output_paths = [os.path.join(scores_dir, f'{scene}.npz') for scene in scenes]

            proposals, losses, scores = self.model(rgbsigma, objectness_output_paths=output_paths)
            proposals_list.extend(proposals)
            scores_list.extend(scores)
            gt_boxes_list.extend(gt_boxes)
            scenes_list.extend(scenes)

        proposals_list = [item.cpu() for item in proposals_list]
        scores_list = [item.cpu() for item in scores_list]
        if not self.args.save_level_index:
            proposals_list = [item[..., 1:] for item in proposals_list]
        if self.args.output_proposals:
            self.output_proposals(scenes_list, proposals_list, scores_list, gt_boxes_list)
            
        if self.args.save_level_index:
            proposals_list = [item[..., 1:] for item in proposals_list]
            
        if gt_boxes_list[0] is None:
            return None, None

        recalls = []
        APs = []
        json_dict = {}

        for limit in [300, 1000, self.args.fpn_post_nms_top_n]:
            if limit > self.args.fpn_post_nms_top_n:
                continue
            # Recalls
            recall50 = evaluate_box_proposals_recall(proposals_list, scores_list, gt_boxes_list, 
                                                     thresholds=torch.tensor([0.5]), limit=limit)
            recall25 = evaluate_box_proposals_recall(proposals_list, scores_list, gt_boxes_list, 
                                                     thresholds=torch.tensor([0.25]), limit=limit)

            ar = evaluate_box_proposals_recall(proposals_list, scores_list, gt_boxes_list,
                                               thresholds=torch.arange(0.25, 1.0, 0.05), limit=limit)

            recalls.append(recall50['ar'].item())
            json_dict[f'recall_50_top_{limit}'] = recall50
            json_dict[f'recall_25_top_{limit}'] = recall25
            json_dict[f'recall_ar_top_{limit}'] = ar

            print(f'\nTop {limit} proposals:')
            print(f'Recall@50: Recall: {recall50["ar"].item():.4f}, Num pos: {recall50["num_pos"]}')
            print(f'Recall@25: Recall: {recall25["ar"].item():.4f}, Num pos: {recall25["num_pos"]}')
            print(f'AR: {ar["ar"].item():.4f}')

            if self.args.wandb:
                wandb.log({
                    f'Recall50 top{limit}': recall50['ar'].item(),
                    f'Recall25 top{limit}': recall25['ar'].item(),
                    f'AR top{limit}': ar['ar'].item(),
                }, commit=False)

        # Average precisions
        ap50 = evaluate_box_proposals_ap(proposals_list, scores_list, gt_boxes_list, iou_thresh=0.5,
                                         top_k=self.args.ap_top_n)
        ap25 = evaluate_box_proposals_ap(proposals_list, scores_list, gt_boxes_list, iou_thresh=0.25,
                                         top_k=self.args.ap_top_n)

        APs.append(ap50['ap'].item())

        print(f'AP@50: AP: {ap50["ap"].item():.4f}')
        print(f'AP@25: AP: {ap25["ap"].item():.4f}')

        json_dict['ap_50'] = ap50
        json_dict['ap_25'] = ap25

        if self.args.mode == 'eval':
            for metric in json_dict:
                for item in json_dict[metric]:
                    if isinstance(json_dict[metric][item], torch.Tensor):
                        json_dict[metric][item] = json_dict[metric][item].tolist()

            os.makedirs(self.args.save_path, exist_ok=True)
            with open(os.path.join(self.args.save_path, 'eval.json'), 'w') as f:
                json.dump(json_dict, f, indent=2)

        if self.args.wandb:
            wandb.log({
                f'ap50': ap50['ap'].item(),
                f'ap25': ap25['ap'].item(),
            }, commit=True)

        return recalls, APs

    # https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    @torch.no_grad()
    def benchmark(self):
        dummy_input = [torch.randn(4, 160, 160, 160, dtype=torch.float).cuda()]
        self.model.eval()

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        for rep in tqdm(range(repetitions)):
            starter.record()
            _ = self.model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(f'Average inference time: {mean_syn:.4f} ms, std: {std_syn:.4f} ms')


def main_worker(proc, nprocs, args, gpu_ids):
    '''
    Main worker function for multiprocessing.
    '''
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:17669', world_size=nprocs, rank=proc)
    torch.cuda.set_device(gpu_ids[proc])

    logger = logging.getLogger(f'worker_{proc}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if args.log_to_file:
        log_dir = os.path.join(args.save_path, 'log')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, f'worker_{proc}.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    trainer = Trainer(args, proc, nprocs, gpu_ids[proc], logger)
    dist.barrier()
    if args.mode == 'train':
        trainer.train_loop()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')

    gpu_ids = []
    if args.gpus:
        for token in args.gpus.split(','):
            if '-' in token:
                start, end = token.split('-')
                gpu_ids.extend(range(int(start), int(end)+1))
            else:
                gpu_ids.append(int(token))

    if len(gpu_ids) <= 1:
        if len(gpu_ids) == 1:
            torch.cuda.set_device(gpu_ids[0])

        logger = None
        if args.log_to_file:
            log_dir = os.path.join(args.save_path, 'log')
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, f'worker_0.log'))
            file_handler.setFormatter(logging.Formatter('[%(asctime)s %(levelname)s] %(message)s'))
            file_handler.setLevel(logging.DEBUG)
            logger = logging.getLogger('worker_0')
            logger.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)

        trainer = Trainer(args, logger=logger)

        if args.mode == 'train':
            trainer.train_loop()
        elif args.mode == 'eval':
            trainer.eval(trainer.test_set)
            # trainer.eval(trainer.val_set)

        elif args.mode == 'benchmark':
            trainer.benchmark()
    else:
        nprocs = len(gpu_ids)
        logging.info(f'Using {nprocs} processes for DDP, GPUs: {gpu_ids}')
        mp.spawn(main_worker, nprocs=nprocs, args=(nprocs, args, gpu_ids), join=True)


if __name__ == '__main__':
    main()

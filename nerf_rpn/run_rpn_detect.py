import os
import glob
import math
import json
import torch
import numpy as np
import argparse
import logging

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.feature_extractor import Bottleneck, ResNet_FPN_256, ResNet_FPN_64
from model.feature_extractor import VGG_FPN, SwinTransformer_FPN
from model.detector import ProposalTargetLayer, ROIPool, RCNN, Classification_Model
from model.utils import clip_boxes_to_mesh, remove_small_boxes, nms
from datasets import RPNClassificationDataset
from eval import evaluate_box_proposals_ap

from tqdm import tqdm
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Train and eval the NeRF RPN baseline on 3D FRONT.')

    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--debug_mode', action='store_true', help='Turn to debug mode.')

    parser.add_argument('--features_path', default='', help='The path to the features.')
    parser.add_argument('--boxes_path', default='', help='The path to the boxes.')
    parser.add_argument('--rois_path', default='', help='The path to the rois.')

    parser.add_argument('--save_root', default='', help='The root to save the model.')
    parser.add_argument('--save_path', default='', help='The path to save the model. It will create a folder named the process_name under save_root')
    parser.add_argument('--dataset_split', default='', help='The dataset split to use.')
    parser.add_argument('--checkpoint', default='', help='The path to the checkpoint to load.')
    parser.add_argument('--pretrained', default='', help='The path to the pretrained backbone to load.')

    parser.add_argument('--bash_file', default='', help='The bash to run the code.')
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune the backbone.')
    parser.add_argument('--backbone_type', type=str, default='resnet', choices=['resnet', 'vgg_AF', 'vgg_EF', 'swin'], 
                        help='The backbone type to use.')
    parser.add_argument('--backbone_input_dim', type=int, default=4, help='Input dimension for backbone.')
    parser.add_argument('--resolution', type=int, default=160, help='The max resolution of the input features.')
    parser.add_argument('--normalize_density', action='store_true', help='Whether to normalize the density.')
    parser.add_argument('--output_proposals', action='store_true', 
                        help='Whether to output proposals during evaluation.')
    parser.add_argument('--filter', choices=['none', 'tp', 'fp'], default='none', 
                        help='Filter for the proposal output.')
    parser.add_argument('--filter_threshold', type=float, default=0.5,
                        help='The IoU threshold for the proposal filter, only used if --output_proposals is True '
                        'and --filter is not "none".')

    # Training parameters
    parser.add_argument('--batch_size', default=2, type=int, help='The num of scenes in a batch.')
    parser.add_argument('--num_epochs', default=100, type=int, help='The number of epochs to train.')
    parser.add_argument('--lr', default=1e-4, type=float, help='The learning rate.')
    parser.add_argument('--reg_loss_weight', default=5.0, type=float, 
                        help='The weight for balancing the regression loss.')
    parser.add_argument('--weight_decay', default=0.0005, type=float, 
                        help='The weight decay strength (L2 regularization).')
    parser.add_argument('--clip_grad_norm', default=0.1, type=float, help='The gradient clipping norm.')

    parser.add_argument('--rotate_prob', default=0.5, type=float, help='The probability of rotating the scene.')
    parser.add_argument('--flip_prob', default=0.5, type=float, help='The probability of flipping the scene.')
    parser.add_argument('--rot_scale_prob', default=0.5, type=float, help='The probability of extra scaling and rotation.')

    parser.add_argument('--log_interval', default=20, type=int, help='The number of iterations to print the loss.')
    parser.add_argument('--eval_interval', default=1, type=int, help='The number of epochs to evaluate.')
    parser.add_argument('--keep_checkpoints', default=1, type=int, help='The number of latest checkpoints to keep.')
    parser.add_argument('--wandb', action='store_true', help='Whether to use wandb for logging.')
    parser.add_argument('--process_root', default='wandb_root', type=str, help='The root of the process in wandb and model save.')
    parser.add_argument('--process_name', default='wandb_process', type=str, help='The name of the process.')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:23441',  type=str, help='init method of wandb.')

    # Distributed training parameters
    parser.add_argument('--gpus', default='', help='The gpus to use for distributed training. If empty, '
                        'uses the first available gpu. DDP is only enabled if this is greater than one.')

    # cls parameters
    parser.add_argument('--n_classes', default=2, type=int, 
                        help='The number of output classes.')
    parser.add_argument('--output_size', nargs='+', type=int)
    parser.add_argument('--spatial_scale', nargs='+', type=int)
    parser.add_argument('--feature_input_dim', default=256, type=int, 
                        help='The input dimension of features.')   
    parser.add_argument('--obj_only', action='store_true',
                        help='Only use objectness loss, i.e. only do classification if it is true.')
    parser.add_argument('--enlarge_scale', default=0.2, type=float, 
                        help='The scale to enlarge the bounding box when sampling the features in feature pooling' )
    parser.add_argument('--use_cuda', action='store_true', 
                        help='Use cuda version of RoI alignment if it is true.')  
    parser.add_argument('--remap', action='store_true', 
                        help='Use FPN version of alignment mapping.')
    parser.add_argument('--is_add_layer', action='store_true', 
                        help='Add an extra convolution layer if it is true.')
    parser.add_argument('--feature_extracting_type', default='pooling', choices=['pooling', 'interpolation'])
    
    parser.add_argument('--nms_thresh', default=0.1, type=float,
                        help='The NMS threshold.')
    parser.add_argument('--filter_score_threhold', default=0.5, type=float,
                        help='The classification score threshold to do nms.')    
    parser.add_argument('--filter_num_threhold', default=300, type=float,
                        help='The maximum number of rois to do nms.')    
    
    parser.add_argument('--cls_batch_size', default=512, type=int, 
                        help='The number of rois sampled from each scene.')
    parser.add_argument('--fg_fraction', default=0.5, type=float, 
                        help='The fraction of fg objections when sampling.')   
    parser.add_argument('--fg_threshold', default=0.35, type=float, 
                        help='The threshold which determines whether the label of a roi is fg.')   
    parser.add_argument('--bg_threshold', default=0.15, type=float, 
                        help='The threshold which determines whether the label of a roi is bg.')  
    parser.add_argument('--top_k', default = None, type=int,
                        help='The proposals with the top k scores will be used to calculate AP ')
    parser.add_argument('--rotated_bbox', action='store_true', 
                        help='If yes, bbox: (N, 7), [x, y, z, w, h, d, theta] \
                              If no, bbox: (N, 6), [xmin, ymin, zmin, xmax, ymax, zmax]')
    parser.add_argument('--is_flatten', action='store_true', 
                        help='Whether to flatten the features from pooling')
    parser.add_argument('--log_to_file', action='store_true', 
                        help='Whether to log output')  

    parser.add_argument('--output_all',  action='store_true')

    args = parser.parse_args()
    return args


class Trainer:
    def __init__(self, args, rank=0, world_size=1, device_id=None, logger=None):
        self.args = args
        self.rank = rank
        self.world_size = world_size

        self.device_id = device_id
        self.logger = logger if logger is not None else logging.getLogger()
        if logger is None:
            self.logger.setLevel(logging.INFO)
        if args.wandb and rank == 0:
            project_root = "nerf_objectness_debug" if args.debug_mode \
                            else "nerf_objectness_" + self.args.process_root
            wandb.init(project=project_root)
            wandb.run.name = args.process_name
            wandb.config.update(args)

        self.logger.info('Constructing model.')

        block = Bottleneck

        self.backbone = None
        if args.fine_tune:
            self.build_backbone()
        
        self.sample_model = ProposalTargetLayer(args.n_classes, batch_size=args.cls_batch_size, 
                                                fg_fraction=args.fg_fraction, 
                                                fg_threshold=args.fg_threshold, 
                                                bg_threshold=args.bg_threshold,
                                                is_rotated_bbox=args.rotated_bbox)
        self.pooling_model = ROIPool(args.output_size, args.spatial_scale, args.enlarge_scale, 
                                     is_rotated_bbox=args.rotated_bbox, 
                                     feature_extracting_type=args.feature_extracting_type, 
                                     max_res=args.resolution)
        self.RCNN_model = RCNN(args.feature_input_dim, block, args.n_classes, args.output_size, 
                               is_add_layer=args.is_add_layer, is_rotated_bbox=args.rotated_bbox, 
                               is_flatten=args.is_flatten)
        
        self.min_size = 1e-3

        if args.pretrained and args.fine_tune:
            assert os.path.exists(args.pretrained), 'The pretrained model does not exist.'
            self.logger.info(f'Loading pretrained backbone from {args.pretrained}.')
            checkpoint = torch.load(args.pretrained)
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])

        if args.checkpoint:
            assert os.path.exists(args.checkpoint), 'The checkpoint does not exist.'
            self.logger.info(f'Loading checkpoint from {args.checkpoint}.')
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            print('Training args from checkpoint:')
            print(checkpoint['train_args'])

            self.RCNN_model.load_state_dict(checkpoint['RCNN_dict'])
            if self.backbone is not None:
                self.backbone.load_state_dict(checkpoint['backbone_state_dict'])

        self.model = Classification_Model(
            self.backbone,
            self.sample_model,
            self.pooling_model,
            self.RCNN_model,
            # spatial_scale = args.spatial_scale,
            n_classes=args.n_classes,
            is_training=True if args.mode == 'train' else False,
            batch_size=args.batch_size,
            is_rotated_bbox=args.rotated_bbox
        )

        if torch.cuda.is_available():
            self.model.cuda()

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device_id], find_unused_parameters=True)
        self.init_datasets()


    def init_datasets(self):
        if not self.args.dataset_split:
            raise ValueError('The dataset split must be specified.')

        self.logger.info(f'Loading dataset split from {self.args.dataset_split}.')

        with np.load(self.args.dataset_split) as split:
            self.train_scenes = split['train_scenes']
            self.test_scenes = split['test_scenes']
            self.val_scenes = split['val_scenes']
            if self.args.output_all:
                self.test_scenes = np.concatenate([self.train_scenes, self.test_scenes, self.val_scenes])

        if self.args.mode == 'eval':
            self.test_set = RPNClassificationDataset(self.args.features_path, self.args.boxes_path, self.args.rois_path,
                                                    scene_names=self.test_scenes, fine_tune=self.args.fine_tune,
                                                    normalize_density=self.args.normalize_density)
            if self.rank == 0:
                self.logger.info(f'{len(self.test_set)} testing scenes, ')
        else:
            self.val_set = RPNClassificationDataset(self.args.features_path, self.args.boxes_path, self.args.rois_path,
                                                    scene_names=self.val_scenes, fine_tune=self.args.fine_tune,
                                                    normalize_density=self.args.normalize_density)
            if self.rank == 0:
                self.logger.info(f'{len(self.val_set)} validation scenes.')

    def build_backbone(self):
        if self.args.backbone_type == 'resnet':
            if not self.args.simple_backbone:
                if self.args.resolution == 64:
                    self.backbone = ResNet_FPN_64(Bottleneck, [3, 4, 6, 3], input_dim=self.args.backbone_input_dim, use_fpn=True)
                else:
                    self.backbone = ResNet_FPN_256(Bottleneck, [3, 4, 6, 3], input_dim=self.args.backbone_input_dim, is_max_pool=True)
        elif self.args.backbone_type == 'vgg_AF':
            self.backbone = VGG_FPN("AF", self.args.backbone_input_dim, True, self.args.resolution)
        elif self.args.backbone_type == 'vgg_EF':
            self.backbone = VGG_FPN("EF", self.args.backbone_input_dim, True, self.args.resolution)
        elif self.args.backbone_type == 'swin':
            # only for 256 features
            self.backbone = SwinTransformer_FPN(patch_size=[4, 4, 4], 
                                                embed_dim=96, 
                                                depths=[2, 2, 18, 2], 
                                                num_heads=[3, 6, 12, 24], 
                                                window_size=[4, 4, 4],
                                                stochastic_depth_prob=0,
                                                expand_dim=True,
                                                input_dim=self.args.backbone_input_dim)

    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'backbone_state_dict': self.backbone.state_dict() if self.args.fine_tune else None,
            'RCNN_dict': self.RCNN_model.state_dict(),
            'train_args': self.args.__dict__,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, path)

    def delete_old_checkpoints(self, path, keep_latest=5):
        files = glob.glob(f'{path}/epoch_*.pt')
        files.sort(key=os.path.getmtime)
        if len(files) > keep_latest:
            for file in files[:-keep_latest]:
                logging.info(f'Deleting old checkpoint {file}.')
                os.remove(file)

    def train_loop(self):
        self.train_set = RPNClassificationDataset(self.args.features_path, self.args.boxes_path, self.args.rois_path,
                                                  scene_names=self.train_scenes, fine_tune=self.args.fine_tune,
                                                  normalize_density=self.args.normalize_density, 
                                                  rotate_prob=self.args.rotate_prob,
                                                  flip_prob=self.args.flip_prob,
                                                  rotate_scale_prob=self.args.rot_scale_prob)
        if self.world_size == 1:
            self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, 
                                           collate_fn=RPNClassificationDataset.collate_fn,
                                           shuffle=True, num_workers=4, pin_memory=True)
        else:
            self.train_sampler = DistributedSampler(self.train_set)
            self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size // self.world_size,
                                           collate_fn=RPNClassificationDataset.collate_fn,
                                           sampler=self.train_sampler, num_workers=2, pin_memory=True)
       
        if self.rank == 0:
            self.logger.info(f'{len(self.train_set)} training scenes.')
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, 
                            weight_decay=self.args.weight_decay)
        self.scheduler = OneCycleLR(self.optimizer, max_lr=self.args.lr,
                                    total_steps=self.args.num_epochs * len(self.train_loader))
        self.best_metric = None
        
        start_epoch = 0
        if self.args.checkpoint:
            checkpoint = torch.load(self.args.checkpoint,  map_location='cpu')
            if 'optimizer_state_dict' in checkpoint.keys():
                self.logger.info(f'Loading optimizer from {self.args.pretrained}')
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint.keys():
                self.logger.info(f'Loading scheduler from {self.args.pretrained}')
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch']
        os.makedirs(self.args.save_path, exist_ok=True)
        # self.eval(self.val_set)
        
        for epoch in range(start_epoch, start_epoch + self.args.num_epochs):
            if self.world_size > 1:
                self.train_sampler.set_epoch(epoch)

            self.train_epoch(epoch)
            if self.rank != 0 :
                continue
            if epoch % self.args.eval_interval == 0 or epoch == self.args.num_epochs - 1:
                APs, precisions, accs = self.eval(self.val_set)
                metric = APs     # AP25

                if self.best_metric is None:
                    self.best_metric = metric
                else:
                    for metric_name in list(metric.keys()):
                        if metric[metric_name] > self.best_metric[metric_name]:
                            self.best_metric[metric_name] = metric[metric_name]
                            self.save_checkpoint(epoch, os.path.join(self.args.save_path, f'model_best_ap{metric_name}.pt'))

                self.save_checkpoint(epoch, os.path.join(self.args.save_path, f'epoch_{epoch}.pt'))
                self.delete_old_checkpoints(self.args.save_path, keep_latest=self.args.keep_checkpoints)

    def train_epoch(self, epoch):
        for i, batch in enumerate(self.train_loader):
            self.logger.info(f'GPU {self.device_id} {batch[-1]}')
            self.model.train()
            self.optimizer.zero_grad()

            level_features, boxes, rois, scene_name = batch

            # Since we do a sample binary classification, the gt values of all gt boxes are 1
            gt_labels = [i.new_ones(i.size(0)) for i in boxes]
            if torch.cuda.is_available():
                level_features = [[i.cuda() for i in item] for item in level_features]
                boxes = [item.cuda() for item in boxes]
                rois = [item.cuda() for item in rois]

            proposals, scores, losses  = self.model(rois, boxes, gt_labels, level_features)
            
            losses['loss_rpn_box_reg'] *= self.args.reg_loss_weight
            if self.args.obj_only:
                loss = losses['loss_objectness'] 
            else:
                loss = losses['loss_objectness'] + losses['loss_rpn_box_reg']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            if self.world_size > 1:
                dist.barrier()
                dist.all_reduce(losses['loss_objectness'])
                dist.all_reduce(losses['loss_rpn_box_reg'])
                dist.all_reduce(loss)

                losses['loss_objectness'] /= self.world_size
                losses['loss_rpn_box_reg'] /= self.world_size
                loss /= self.world_size
            if i % self.args.log_interval == 0 and self.rank == 0:
                self.logger.info(f'Epoch {epoch} [{i}/{len(self.train_loader)}]  '
                             f'Loss: {loss.item():.4f}  '
                             f'Obj loss: {losses["loss_objectness"].item():.4f}  '
                             f'Reg loss: {losses["loss_rpn_box_reg"].item():.4f}')
                        
            if self.args.wandb and self.rank == 0:
                wandb.log({
                    'lr': self.scheduler.get_last_lr()[0],
                    'loss': loss.item(),
                    'objectness_loss': losses['loss_objectness'].item(),
                    'regression_loss': losses['loss_rpn_box_reg'].item()
                })

    def output_proposals(self, scenes, proposals, scores, gt_boxes, threshold = 0.7):
        output_path = os.path.join(self.args.save_path, 'objectness')
        os.makedirs(output_path, exist_ok=True)
        for scene, proposal, score, gt in zip(scenes, proposals, scores, gt_boxes):
            keep = torch.nonzero(score >= threshold)
            score = score[keep]
            proposal = proposal[keep]
            threshold_name = math.floor(threshold * 1000)
            os.makedirs(os.path.join(output_path, f'{threshold_name}'), exist_ok=True)
            np.savez(os.path.join(output_path, f'{threshold_name}', f'{scene}.npz'), proposal=proposal, score=score)

    @torch.no_grad()
    def filter_proposals(
        self,
        proposals,
        objectness,
        gt_labels,
        mesh_sizes,
        score_threhold=0.8
    ):
        final_boxes = []
        final_scores = []
        final_gt_labels = []
        bbox_size = 7 if self.args.rotated_bbox else 6
        for boxes, scores, gt_labels, mesh_shape in zip(proposals, objectness, gt_labels, mesh_sizes):

            boxes = clip_boxes_to_mesh(boxes, mesh_shape)
            # remove small boxes
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, scores, gt_labels = boxes[keep], scores[keep], gt_labels[keep]
            
            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= score_threhold)[0]
            boxes, scores, gt_labels = boxes[keep], scores[keep], gt_labels[keep]

            # non-maximum suppression, independently done per level
            keep = nms(boxes[..., :bbox_size], scores[..., 1], self.args.nms_thresh)
            keep = keep[scores[..., 1][keep].sort(descending=True)[1]]

            # keep only topk scoring predictions
            boxes, scores, gt_labels = boxes[keep], scores[keep], gt_labels[keep]

            final_boxes.append(boxes[..., :bbox_size])
            final_scores.append(scores)
            final_gt_labels.append(gt_labels)

        return final_boxes, final_scores, final_gt_labels

    @torch.no_grad()
    def eval(self, dataset):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, 
                                shuffle=False, num_workers=1,
                                collate_fn=RPNClassificationDataset.collate_fn)

        self.logger.info(f'Evaluating...')

        proposals_list = []
        scores_list = []
        gt_label_list = []
        gt_boxes_list = []
        scenes_list = []

        for batch in tqdm(dataloader):
            level_features, boxes, rois, scene_name = batch
            gt_labels = [i.new_ones(i.size(0)) for i in boxes]
            rois_split =[]
            max_batch_size = self.args.cls_batch_size // self.world_size
            # max_batch_size = 256
            for items in rois:
                batch_split = []
                size = items.size(0)
                for s in range(math.ceil(size / max_batch_size) + 1):
                    batch_split.append(min(s * max_batch_size, size))
                rois_split.append(batch_split)
            
            proposals, scores = [[],[]], []

            if torch.cuda.is_available():
                level_features = [[i.cuda() for i in item] for item in level_features]
                boxes = [item.cuda() for item in boxes]
                rois = [item.cuda() for item in rois]
            
            for b in range(len(rois_split)):
                c_rois = rois[b]
                c_split = rois_split[b].copy()
                temp0, temp1, temp2 = [], [], []
            
                for s in range(len(c_split)-1):

                    c_proposals, c_scores, losses, = self.model([c_rois[c_split[s]:c_split[s+1]]], boxes, gt_labels, 
                        level_features, is_sample=False, is_reg=not self.args.obj_only)
                    temp0.extend([item.cpu() for item in c_proposals[0]])
                    temp1.extend([item.cpu() for item in c_proposals[1]])
                    temp2.extend([item.cpu() for item in c_scores])

                proposals[0].append(torch.cat(temp0))
                proposals[1].append(torch.cat(temp1))
                scores.append(torch.cat(temp2))

            mesh_sizes = [level_features[iter][0].shape[1:] for iter in range(len(level_features))]
            mesh_sizes = [[iter_iter * self.args.spatial_scale[0] for iter_iter in iter] for iter in mesh_sizes]

            proposals, scores, gt_labels = self.filter_proposals(proposals[0], scores, proposals[1], mesh_sizes, 
                                                                 score_threhold=self.args.filter_score_threhold)

            boxes = [item.cpu() for item in boxes]
            gt_labels = [item.cpu() for item in gt_labels]
            
            level_features = [[i.cpu() for i in item] for item in level_features]
            boxes = [item.cpu() for item in boxes]
            rois = [item.cpu() for item in rois]
            torch.cuda.empty_cache()
            proposals_list.extend(proposals)
            gt_label_list.extend(gt_labels)
            scores_list.extend(scores)
            gt_boxes_list.extend(boxes)
            scenes_list.extend(scene_name)

        proposals_list = [item.cpu() for item in proposals_list]
        scores_list = [item[..., 1].cpu() for item in scores_list]
        gt_boxes_list = [item.cpu() for item in gt_boxes_list]

        precisions = []
        accuracy = []
        json_dict = {}

        APs = {}
        for iou_thresh in [0.25, 0.5]:
            AP = evaluate_box_proposals_ap(proposals_list, scores_list, gt_boxes_list, iou_thresh, top_k=self.args.top_k)

            print(f'AP{int(iou_thresh * 100)}: {AP["ap"].item()}')

            # print(AP)
            if self.args.wandb and self.rank == 0:
                wandb.log({
                    f'AP{int(iou_thresh * 100)}': AP['ap'].item(),
                }, commit=False)

            json_dict[f'AP{int(iou_thresh * 100)}'] = AP

            APs[f'{int(iou_thresh * 100)}'] = AP['ap'].item()

        if self.args.mode == 'eval':
            for metric in json_dict:
                for item in json_dict[metric]:
                    if isinstance(json_dict[metric][item], torch.Tensor):
                        json_dict[metric][item] = json_dict[metric][item].tolist()
            # eval_file = os.path.join(self.args.save_path, 'eval.json')
            
            eval_file = os.path.join(self.args.save_path, 'eval.json')
            mode = 'a' if os.path.exists(eval_file) else 'w'
            with open(eval_file, mode) as f:
                json.dump(json_dict, f, indent=2)
                
        if self.args.wandb:
            wandb.log({}, commit=True)

        return APs, precisions, accuracy


def main_worker(proc, nprocs, args, gpu_ids):
    '''
    Main worker function for multiprocessing.
    '''
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=nprocs, rank=proc)
    torch.cuda.set_device(gpu_ids[proc])

    logger = logging.getLogger(f'worker_{proc}')
    logger.setLevel(logging.INFO)
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
    if args.debug_mode:
        args.process_name = "debug_" + args.process_name 
        
    if args.save_root != '':
        os.makedirs(args.save_root, exist_ok=True)
    if args.process_root != '':
        args.save_root = os.path.join(args.save_root, args.process_root)
    if args.process_name != '':
        args.save_path = os.path.join(args.save_root, args.process_name)
        os.makedirs(args.save_path, exist_ok=True)

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
        trainer = Trainer(args)
        if args.mode == 'train':
            trainer.train_loop()
        elif args.mode == 'eval':
            trainer.eval(trainer.test_set)
    else:
        nprocs = len(gpu_ids)
        logging.info(f'Using {nprocs} processes for DDP, GPUs: {gpu_ids}')
        mp.spawn(main_worker, nprocs=nprocs, args=(nprocs, args, gpu_ids), join=True)


if __name__ == '__main__':
    main()

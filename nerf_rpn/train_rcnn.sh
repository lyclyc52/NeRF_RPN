#!/usr/bin/env bash
set -x
set -e

DATA_ROOT=/data/hypersim_rpn_data

python3 -u run_rpn_detect.py \
--mode train \
--features_path ${DATA_ROOT}/features \
--boxes_path ${DATA_ROOT}/obb \
--dataset_split ${DATA_ROOT}/hypersim_split_new.npz \
--rois_path ./results/hypersim_anchor_swinS/proposals \
--pretrained ./results/hypersim_anchor_swinS/model_best.pt \
--save_root ./results/objectness_model \
--fine_tune \
--backbone_type swin \
--num_epochs 1000 \
--lr 1e-4 \
--reg_loss_weight 5. \
--weight_decay 1e-4 \
--log_interval 5 \
--eval_interval 10 \
--keep_checkpoints 5 \
--n_classes 2 \
--output_size 3 3 3 \
--spatial_scale 4 8 16 32 \
--process_root front3d \
--process_name anchor_swin_ft \
--is_add_layer \
--is_flatten \
--rotated_bbox \
--cls_batch_size 256 \
--batch_size 2 \
--fg_threshold 0.25 \
--bg_threshold 0.25 \
--backbone_input_dim 4 \
--resolution 200 \
--gpus 0
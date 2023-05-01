#!/usr/bin/env bash

set -x
set -e

DATA_ROOT=/data/front3d_rpn_data

python3 -u run_rpn.py \
--mode train \
--dataset_name front3d \
--resolution 160 \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--boxes_path ${DATA_ROOT}/obb \
--dataset_split ${DATA_ROOT}/3dfront_split.npz \
--save_path ./results/front3d_anchor_swins \
--num_epochs 200 \
--lr 3e-4 \
--weight_decay 1e-3 \
--log_interval 10 \
--eval_interval 10 \
--rpn_nms_thresh 0.3 \
--log_to_file \
--normalize_density \
--rotated_bbox \
--batch_size 8 \
--gpus 0-3

#!/usr/bin/env bash

set -x
set -e

DATA_ROOT=/data/front3d_rpn_data

python3 -u run_fcos.py \
--mode "eval" \
--dataset front3d \
--resolution 160 \
--backbone_type swin_s \
--features_path ${DATA_ROOT}/features \
--boxes_path ${DATA_ROOT}/obb \
--dataset_split ${DATA_ROOT}/3dfront_split.npz \
--save_path ./results/front3d_fcos_swin \
--checkpoint ./results/front3d_fcos_swin/model_best.pt \
--norm_reg_targets \
--centerness_on_reg \
--nms_thresh 0.3 \
--output_proposals \
--save_level_index \
--normalize_density \
--rotated_bbox \
--batch_size 2 \
--gpus 0

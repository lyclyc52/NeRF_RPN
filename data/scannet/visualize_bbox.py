import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import h5py
import random
import argparse
from tqdm.contrib.concurrent import process_map


def get_xform(dist2m):
    # run_colmap.py: y_down to z_up
    xform1 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    # colmap_reader.cpp: scale by dist2m
    xform2 = np.eye(3) * dist2m

    # colmap_reader.cpp: convert from colmap to nerf coordinate
    xform3 = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    return xform3 @ xform2 @ xform1


def get_bbox_corners(obj_dict, dist2m):
    xform = get_xform(dist2m)
    max_pt = np.array(obj_dict['max_pt'])
    min_pt = np.array(obj_dict['min_pt'])

    corners = np.array([
        max_pt,
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        min_pt,
        [min_pt[0], min_pt[1], max_pt[2]]
    ])

    # corners = xform @ corners.T
    return corners


def get_projected_points(world2proj, corners):
    corners = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    projected = (world2proj @ corners.T).T

    keep = (projected[:, 3] > 0)
    projected = projected[keep]
    projected = projected[:, :2] / projected[:, 3][:, np.newaxis]
    projected = projected.astype(np.int32)

    return projected


def get_world_to_proj_matrix(frame_dict, width, height):
    cam2world = np.array(frame_dict['transform_matrix'])
    cam2world[:, [1, 2]] *= -1  # fron nerf to opencv

    fx = frame_dict['fx']
    fy = frame_dict['fy']
    cx = frame_dict['cx']
    cy = frame_dict['cy']

    focal = fy / height
    zscale = 1.0 / focal
    xyscale = height

    cam2proj = np.array([
        [xyscale, 0, width * 0.5 * zscale, 0],
        [0, xyscale, height * 0.5 * zscale, 0],
        [0, 0, 1, 0],
        [0, 0, zscale, 0]
    ])

    # cam2proj = np.array([
    #     [fx, 0, cx, 0],
    #     [0, fy, cy, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 1, 0]
    # ])

    world2cam = np.linalg.inv(cam2world)
    # world2cam[2, :] *= -1
    world2proj = cam2proj @ world2cam

    return world2proj


def get_obb_corners(obj_dict, dist2m):
    xform = get_xform(dist2m)
    obb = np.array(obj_dict['obb'])
    
    corners = np.array([
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
        [1, -1, 1]
    ]) * 0.5 * obb[3:6][None, :]

    angle = obb[6]
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

    corners = rot @ corners.T
    corners = corners.T + obb[:3][None, :]

    # corners = xform @ corners.T
    return corners


def add_bbox_line(img, world2proj, a, b, color=(0, 0, 255), thickness=2):
    ha = np.array([a[0], a[1], a[2], 1]).reshape(4, 1)
    hb = np.array([b[0], b[1], b[2], 1]).reshape(4, 1)
    ha = np.squeeze(world2proj @ ha).T
    hb = np.squeeze(world2proj @ hb).T

    if ha[3] <= 0 or hb[3] <= 0:
        return

    aa = tuple((ha[:2] / ha[3]).astype(np.int32))
    bb = tuple((hb[:2] / hb[3]).astype(np.int32))

    h, w, c = img.shape
    cv2.line(img, aa, bb, color, thickness)


def render_bbox_overlay(img, world2proj, corners, color=(0, 0, 255), thickness=2):
    for i in range(4):
        add_bbox_line(img, world2proj, corners[i, :], corners[(i + 1) % 4, :], color, thickness)
        add_bbox_line(img, world2proj, corners[i + 4, :], corners[(i + 1) % 4 + 4, :], color, thickness)
        add_bbox_line(img, world2proj, corners[i, :], corners[i + 4, :], color, thickness)


def add_semantic_labels(img, corners, label, world2proj, color):
    projected = get_projected_points(world2proj, corners)
    if projected.shape[0] < 2:
        return img

    keep = (projected[:, 0] >= 0) & (projected[:, 0] < img.shape[1]) & (projected[:, 1] >= 0) & (projected[:, 1] < img.shape[0])
    projected = projected[keep]
    if projected.shape[0] == 0:
        return img

    idx = np.argmin(projected[:, 1])

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    cv2.rectangle(img, (projected[idx, 0], projected[idx, 1] - 20), 
                 (projected[idx, 0] + w, projected[idx, 1]), color, -1)

    cv2.putText(img, label, (projected[idx, 0], projected[idx, 1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
    return img


def process_scene(instances, xform_dict, scene_dir, output_dir, dist2m=1.0):
    instances = instances['instances']

    for frame in xform_dict['frames']:
        file_path = os.path.join(scene_dir, frame['file_path'])
        img_name = os.path.basename(file_path)
        c2w_mat = np.array(frame['transform_matrix'])

        img = cv2.imread(file_path)
        h, w, _ = img.shape
        focal = frame['fy'] / h

        world2proj = get_world_to_proj_matrix(frame, w, h)

        for instance in instances:
            # corners = get_bbox_corners(instance, dist2m)
            corners = get_obb_corners(instance, dist2m)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            render_bbox_overlay(img, world2proj, corners, color)
            img = add_semantic_labels(img, corners, instance['label'], world2proj, color)

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, help='path to instance json directory')
    parser.add_argument('--scene_dir', type=str, help='path to scene nerf directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scenes = os.listdir(args.instance_path)
    for scene in tqdm(scenes):
        scene_name = scene.split('.')[0]

        with open(os.path.join(args.instance_path, f'{scene_name}.json'), 'r') as f:
            instances = json.load(f)

        with open(os.path.join(args.scene_dir, scene_name, 'transforms_train.json'), 'r') as f:
            xform_dict = json.load(f)

        scene_path = os.path.join(args.scene_dir, scene_name)
        output_path = os.path.join(args.output_dir, scene_name)
        os.makedirs(output_path, exist_ok=True)

        process_scene(instances, xform_dict, scene_path, output_path)

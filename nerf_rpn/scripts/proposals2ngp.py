import os
import json
import numpy as np
import argparse

from copy import deepcopy
from tqdm import tqdm


def ngp_matrix_to_nerf(ngp_matrix, scale, offset, from_mitsuba):
    result = deepcopy(ngp_matrix)
    if from_mitsuba:
        result[:, [0, 2]] *= -1
    else:
        # Cycle axes xyz->yzx
        result = result[[2, 0, 1], :]
    
    result[:, [1, 2]] *= -1
    result[:, 3] = (result[:, 3] - offset) / scale
    return result


def proposals_to_ngp_boxes(proposals, features_dict):
    grid_res = features_dict['resolution']
    bbox_min = features_dict['bbox_min']
    bbox_max = features_dict['bbox_max']
    scale = features_dict['scale']
    offset = features_dict['offset']
    from_mitsuba = features_dict['from_mitsuba']

    # From z up to y up
    perm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    diag = bbox_max - bbox_min
    box_min = proposals[:, :3] / grid_res * diag + bbox_min
    box_max = proposals[:, 3:] / grid_res * diag + bbox_min

    offset = perm @ offset  # z up to y up

    boxes = []
    for i in range(box_min.shape[0]):
        center = (box_min[i] + box_max[i]) * 0.5
        extent = (box_max[i] - box_min[i]) / scale

        xform = np.eye(3)
        xform = np.concatenate((xform, np.expand_dims(center, 1)), axis=1)
        xform = perm @ xform    # z up to y up
        xform = ngp_matrix_to_nerf(xform, scale, offset, from_mitsuba)

        boxes.append({
            'orientation': xform[:3, :3].tolist(),
            'position': xform[:3, 3].tolist(),
            'extents': extent.tolist(),
        })

    return boxes


def obb_to_ngp_boxes(proposals, features_dict):
    grid_res = features_dict['resolution']
    bbox_min = features_dict['bbox_min']
    bbox_max = features_dict['bbox_max']
    scale = features_dict['scale']
    offset = features_dict['offset']
    from_mitsuba = features_dict['from_mitsuba']

    # From z up to y up
    perm = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

    diag = bbox_max - bbox_min
    pos = proposals[:, :3] / grid_res * diag + bbox_min
    ext = proposals[:, 3:6] / grid_res * diag / scale
    rot = proposals[:, 6]

    offset = perm @ offset  # z up to y up

    boxes = []
    for i in range(pos.shape[0]):
        xform = np.array([
            [np.cos(rot[i]), -np.sin(rot[i]), 0],
            [np.sin(rot[i]), np.cos(rot[i]), 0],
            [0, 0, 1]
        ])
        xform = np.concatenate((xform, np.expand_dims(pos[i], 1)), axis=1)
        xform = perm @ xform  # z up to y up
        xform = ngp_matrix_to_nerf(xform, scale, offset, from_mitsuba)

        boxes.append({
            'orientation': xform[:3, :3].tolist(),
            'position': xform[:3, 3].tolist(),
            'extents': ext[i].tolist(),
        })
        
    return boxes


def process_scene(args, proposal_path, json_path, feature_path, output_path):
    assert os.path.isfile(json_path)
    assert os.path.isfile(feature_path)
    scene_name = os.path.basename(proposal_path).split('.')[0]

    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    proposals_dict = np.load(proposal_path)
    features_dict = np.load(feature_path)

    scores = proposals_dict['score']
    proposals = proposals_dict['proposal']

    # Filter out proposals with score < threshold
    keep = scores > args.threshold
    scores = scores[keep]
    proposals = proposals[keep]

    # Sort proposals by score
    sorted_indices = np.argsort(scores)[::-1]
    scores = scores[sorted_indices]
    proposals = proposals[sorted_indices]

    # Keep only top k proposals
    if len(scores) > args.top_k:
        scores = scores[:args.top_k]
        proposals = proposals[:args.top_k]

    print(f'{scene_name}: {len(scores)} proposals')

    if args.bbox_format == 'aabb':
        boxes = proposals_to_ngp_boxes(proposals, features_dict)
    elif args.bbox_format == 'obb':
        boxes = obb_to_ngp_boxes(proposals, features_dict)

    for i, box in enumerate(boxes):
        boxes[i]['score'] = scores[i].item()

    json_dict['bounding_boxes'] = boxes
    with open(output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert the RPN proposals to bounding boxes in instant-ngp '
                                     'transforms.json.')

    parser.add_argument('--bbox_format', choices=['aabb', 'obb'], required=True,)
    parser.add_argument('--dataset', type=str, required=True, choices=['hypersim', 'front3d'],
                        help='Dataset name. Must be hypersim or front3d.')
    parser.add_argument('--dataset_path', default='', help='Path to the NeRF scenes.')
    parser.add_argument('--features_path', default='', help='Path to the NeRF features.')
    parser.add_argument('--proposals_path', default='', help='Path to the proposal files.')
    parser.add_argument('--output_dir', default='', help='Path to the output directory.')
    parser.add_argument('--threshold', default=0.5, type=float, help='The threshold for the proposal scores.')
    parser.add_argument('--top_k', default=30, type=int, help='The number of proposals to visualize.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    proposal_files = os.listdir(args.proposals_path)
    proposal_files = [f for f in proposal_files if f.endswith('.npz') 
                      and os.path.isfile(os.path.join(args.proposals_path, f))]

    scene_names = [f.split('.')[0] for f in proposal_files]

    for scene_name in (pbar := tqdm(scene_names)):
        proposal_path = os.path.join(args.proposals_path, scene_name + '.npz')
        json_path = os.path.join(args.dataset_path, scene_name, 'train', 'transforms.json')
        feature_path = os.path.join(args.features_path, scene_name + '.npz')
        output_path = os.path.join(args.output_dir, scene_name + '.json')
        process_scene(args, proposal_path, json_path, feature_path, output_path)

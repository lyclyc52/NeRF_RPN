import os
import random
import argparse
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import zoom

from functools import partial
from tqdm.contrib.concurrent import process_map


def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)


def depth_nerf_density_to_alpha(density):
    activation = np.clip(density, a_min=0, a_max=None)  # original nerf uses relu
    return np.clip(1.0 - np.exp(-activation / 100.0), 0.0, 1.0)


def construct_grid(res):
    res_x, res_y, res_z = res
    x = np.linspace(0, res_x, res_x)
    y = np.linspace(0, res_y, res_y)
    z = np.linspace(0, res_z, res_z)

    scale = res.max()
    x /= scale
    y /= scale
    z /= scale

    # Shift by 0.5 voxel
    x += 0.5 * (1.0 / scale)
    y += 0.5 * (1.0 / scale)
    z += 0.5 * (1.0 / scale)

    grid = []
    for i in range(res_z):
        for j in range(res_y):
            for k in range(res_x):
                grid.append([x[k], y[j], z[i]])

    return np.array(grid)


def write_box_vertex_to_ply(f, box):
    f.write(f'{box[0]} {box[1]} {box[2]} 255 255 255\n')
    f.write(f'{box[0]} {box[4]} {box[2]} 255 255 255\n')
    f.write(f'{box[3]} {box[4]} {box[2]} 255 255 255\n')
    f.write(f'{box[3]} {box[1]} {box[2]} 255 255 255\n')
    f.write(f'{box[0]} {box[1]} {box[5]} 255 255 255\n')
    f.write(f'{box[0]} {box[4]} {box[5]} 255 255 255\n')
    f.write(f'{box[3]} {box[4]} {box[5]} 255 255 255\n')
    f.write(f'{box[3]} {box[1]} {box[5]} 255 255 255\n')


def get_obb_corners(xform, extent):
    corners = np.array([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, -1],
        [-1, -1, 1],
    ], dtype=float).T

    corners *= np.expand_dims(extent, 1) * 0.5
    corners = xform[:, :3] @ corners + xform[:, 3, None]

    return corners


def write_obb_vertex_to_ply(f, obb, needs_y_up=False):
    rot = obb[-1]
    xform = np.array([
        [np.cos(rot), -np.sin(rot), 0, obb[0]],
        [np.sin(rot), np.cos(rot), 0, obb[1]],
        [0, 0, 1, obb[2]],
    ])

    corners = get_obb_corners(xform, obb[3:6])

    if needs_y_up:
        # From z up to y up
        perm = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ]).T

        corners = perm @ corners

    for i in range(8):
        f.write(f'{corners[0][i]:4f} {corners[1][i]:4f} {corners[2][i]:4f} 255 255 255\n')


def write_box_edge_to_ply(f, idx):
    for i in range(3):
        f.write(f'{idx + i} {idx + i + 1}\n')
        f.write(f'{idx + i + 4} {idx + i + 5}\n')
        f.write(f'{idx + i} {idx + i + 4}\n')

    f.write(f'{idx} {idx + 3}\n')
    f.write(f'{idx + 4} {idx + 7}\n')
    f.write(f'{idx + 3} {idx + 7}\n')


def write_alpha_grid_to_ply(f, alpha, grid, threshold=0):
    for i in range(len(grid)):
        if alpha[i] > threshold:
            f.write(f'{grid[i][0]:4f} {grid[i][1]:4f} {grid[i][2]:4f} {alpha[i]} {alpha[i]} {alpha[i]}\n')


def write_colormapped_alpha_grid_to_ply(f, alpha, grid, threshold=0):
    '''
    alpha expected to be in 0-1 range
    '''
    colormap = cm.get_cmap('plasma')
    rgb = (colormap(alpha) * 255).astype(np.uint8)
    for i in range(len(grid)):
        if alpha[i] > threshold:
            f.write(f'{grid[i][0]:4f} {grid[i][1]:4f} {grid[i][2]:4f} {rgb[i][0]} {rgb[i][1]} {rgb[i][2]}\n')


def write_objectness_heatmap_to_ply(f, alpha, score, grid, threshold=0):
    '''
    alpha expected to be in 0-1 range
    '''
    colormap = cm.get_cmap('turbo')
    rgb = (colormap(score) * 255).astype(np.uint8)
    for i in range(len(grid)):
        if alpha[i] > threshold:
            f.write(f'{grid[i][0]:4f} {grid[i][1]:4f} {grid[i][2]:4f} {rgb[i][0]} {rgb[i][1]} {rgb[i][2]}\n')


def write_rgb_to_ply(f, rgb, alpha, grid, threshold=0):
    rgb = (rgb * 255).astype(np.uint8)
    for i in range(len(grid)):
        if alpha[i] > threshold:
            f.write(f'{grid[i][0]:4f} {grid[i][1]:4f} {grid[i][2]:4f} {rgb[i][0]} {rgb[i][1]} {rgb[i][2]}\n')


def get_objectness_grid(data, res):
    acc = np.zeros(res)
    for level in ['0', '1', '2', '3']:
        score = data[level][0]
        score = zoom(score, res / np.array(score.shape), order=3)
        # score = np.clip(score, 0, None)
        # score = np.sqrt(score)  # sqrt to make it more visible
        acc += score

    acc = np.transpose(acc, (2, 1, 0)).reshape(-1)
    acc /= acc.max()
    return acc


def visualize_scene(scene_name, output_dir, feature_dir, box_dir=None, box_format='obb', 
                    objectness_dir=None, alpha_threshold=0.01, transpose_yz=False):
    boxes = None
    if box_dir is not None:
        boxes = np.load(os.path.join(box_dir, scene_name + '.npy'), allow_pickle=True)
    feature = np.load(os.path.join(feature_dir, scene_name + '.npz'), allow_pickle=True)

    res = feature['resolution']
    rgbsigma = feature['rgbsigma']
    # if transpose_yz:
    #     rgbsigma = rgbsigma.reshape((res[2], res[1], res[0], -1))
    #     rgbsigma = np.transpose(rgbsigma, (1, 2, 0, 3))
    #     rgbsigma = rgbsigma.reshape((res[0] * res[1] * res[2], -1))
    #     res = res[[2, 0, 1]]

    rgbsigma = np.transpose(rgbsigma, (2, 1, 0, 3)).reshape(-1, 4)

    scene_box = np.concatenate((np.zeros(3), res))
    scale = 1. / res.max()

    alpha = rgbsigma[:, -1]
    # alpha = depth_nerf_density_to_alpha(alpha)
    alpha = density_to_alpha(alpha)
    # alpha = (alpha * 255).astype(np.uint8)
    num_grid_points = (alpha > alpha_threshold).sum()

    if objectness_dir is not None:
        score = np.load(os.path.join(objectness_dir, scene_name + '_objectness.npz'), allow_pickle=True)
        score = get_objectness_grid(score, res)

    with open(os.path.join(output_dir, scene_name + '.ply'), 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        if boxes is not None:
            f.write(f'element vertex {8 * boxes.shape[0] + 8 + num_grid_points}\n'
                    'property float x\n'
                    'property float y\n'
                    'property float z\n'
                    'property uchar red\n'
                    'property uchar green\n'
                    'property uchar blue\n'
                    f'element edge {12 * boxes.shape[0] + 12}\n'
                    'property int vertex1\n'
                    'property int vertex2\n'
                    'end_header\n\n')
        else:
            f.write(f'element vertex {num_grid_points}\n'
                    'property float x\n'
                    'property float y\n'
                    'property float z\n'
                    'property uchar red\n'
                    'property uchar green\n'
                    'property uchar blue\n'
                    'end_header\n\n')

        if boxes is not None:
            write_box_vertex_to_ply(f, scene_box * scale)
            for i in range(boxes.shape[0]):
                if box_format == 'obb':
                    box = boxes[i]
                    box[:6] *= scale
                    write_obb_vertex_to_ply(f, box)
                else:
                    write_box_vertex_to_ply(f, boxes[i] * scale)

        grid = construct_grid(res)
        # write_alpha_grid_to_ply(f, alpha, grid, threshold=alpha_threshold)

        if objectness_dir is None:
            # write_colormapped_alpha_grid_to_ply(f, alpha, grid, threshold=alpha_threshold)
            write_rgb_to_ply(f, rgbsigma[:, :3], alpha, grid, threshold=alpha_threshold)
        else:
            write_objectness_heatmap_to_ply(f, alpha, score, grid, threshold=alpha_threshold)

        if boxes is not None:
            f.write('\n')
            write_box_edge_to_ply(f, 0)
            for i in range(boxes.shape[0]):
                write_box_edge_to_ply(f, (i + 1) * 8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate ply files of NeRF RPN input features and boxes for visualization.')

    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Path to the directory to save the ply files.')
    parser.add_argument('--feature_dir', '-f', type=str, required=True,
                        help='Path to the directory containing the NeRF RPN input features.')
    parser.add_argument('--box_dir', '-b', type=str, default=None,
                        help='Path to the directory containing the boxes.')
    parser.add_argument('--box_format', '-bf', type=str, default='obb',
                        help='Format of the boxes. Can be either "obb" or "aabb".')
    parser.add_argument('--objectness_dir', type=str, default=None,
                        help='Path to the directory containing the objectness scores.')
    parser.add_argument('--alpha_threshold', type=float, default=0.01,
                        help='Threshold for alpha.')
    parser.add_argument('--transpose_yz', '-tr', action='store_true',
                        help='Whether to transpose the y and z axes.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    npz_files = os.listdir(args.feature_dir)
    npy_files = os.listdir(args.box_dir) if args.box_dir is not None else None

    npz_files = [f for f in npz_files if f.endswith('.npz') and os.path.isfile(os.path.join(args.feature_dir, f))]

    scenes = [f.split('.')[0] for f in npz_files]

    os.makedirs(args.output_dir, exist_ok=True)

    fn = partial(visualize_scene, output_dir=args.output_dir, feature_dir=args.feature_dir, box_dir=args.box_dir,
                 box_format=args.box_format, objectness_dir=args.objectness_dir, alpha_threshold=args.alpha_threshold,
                 transpose_yz=args.transpose_yz)
    
    process_map(fn, scenes, max_workers=8)

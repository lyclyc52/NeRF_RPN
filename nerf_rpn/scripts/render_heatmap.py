# from render_volume_misc import gkern_2d, gkern_3d, obb2hbb, density_to_alpha, world2grid, 
# from scipy.ndimage.filters import gaussian_filter
from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyvista as pv
from pyvista import examples
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import cv2
import os
from os.path import join
from argparse import ArgumentParser
import json
import copy
from scipy.ndimage import gaussian_filter
from bbox_proj import project_obb_to_image

def gkern_3d(w=10, l=10, h=3, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    Reference: https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """
    ax = np.linspace(-(w - 1) / 2., (w - 1) / 2., w)
    ay = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    az = np.linspace(-(h - 1) / 2., (h - 1) / 2., h)
    gauss_x = np.exp(-0.5 * np.square(ax) / np.square(w/5))
    gauss_y = np.exp(-0.5 * np.square(ay) / np.square(l/5))
    gauss_z = np.exp(-0.5 * np.square(az) / np.square(h/5))
    kernel = np.outer(np.outer(gauss_x, gauss_y), gauss_z).reshape(w, l, h)
    return kernel

def obb2point8(obboxes):
    """
    Args:
        obboxes (N, 7): [x, y, z, w, l, h, theta]
    Returns:
        obboxes_8 (N, 8, 3): 8 corners of the obboxes
    """
    x, y, z, w, l, h, theta = np.split(obboxes, [1, 2, 3, 4, 5, 6], axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias1 = w/2 * Cos - l/2 * Sin
    x_bias2 = w/2 * Cos + l/2 * Sin
    y_bias1 = w/2 * Sin + l/2 * Cos
    y_bias2 = w/2 * Sin - l/2 * Cos
    xy1 = np.concatenate([x+x_bias1, y+y_bias1], axis=-1)
    xy2 = np.concatenate([x+x_bias2, y+y_bias2], axis=-1)
    xy3 = np.concatenate([x-x_bias1, y-y_bias1], axis=-1)
    xy4 = np.concatenate([x-x_bias2, y-y_bias2], axis=-1)
    z1, z2 = z-h/2, z+h/2
    return np.concatenate([xy1, z1, xy2, z1, xy3, z1, xy4, z1, 
                           xy1, z2, xy2, z2, xy3, z2, xy4, z2,], axis=-1).reshape(-1, 8, 3)

def aabb2point8(aabbs):
    """
    Args:
        aabbs (N, 6): [x1, y1, z1, x2, y2, z2]
    Returns:
        aabbs_8 (N, 8, 3): 8 corners of the aabbs
    """
    x1, y1, z1, x2, y2, z2 = np.split(aabbs, [1, 2, 3, 4, 5], axis=-1)
    return np.concatenate([x1, y1, z1, x2, y1, z1, x2, y2, z1, x1, y2, z1, 
                           x1, y1, z2, x2, y1, z2, x2, y2, z2, x1, y2, z2], axis=-1).reshape(-1, 8, 3).astype(np.float32)


def obb2hbb(obboxes):
    """Return the smallest 3D AABB that contains the 3D OBB."""
    center, z, w, l, h, theta = np.split(obboxes, [2, 3, 4, 5, 6], axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias = (np.abs(w/2 * Cos) + np.abs(l/2 * Sin))
    y_bias = (np.abs(w/2 * Sin) + np.abs(l/2 * Cos))
    bias = np.concatenate([x_bias, y_bias], axis=-1)
    return np.concatenate([center-bias, z-h/2, center+bias, z+h/2], axis=-1)

def density_to_alpha(density):
    return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)

def world2grid(points, room_bbox, res, downsample=1):
    # points (..., 3)
    # room_bbox [xmin, ymin, zmin, xmax, ymax, zmax]
    points -= room_bbox[:3]
    points /= np.max(room_bbox[3:] - room_bbox[:3])
    points *= np.max(res)
    return points / downsample

def grid2world(points, room_bbox, res):
    points /= np.max(res)
    points *= np.max(room_bbox[3:] - room_bbox[:3])
    points += room_bbox[:3]
    return points

def heatmap_overlap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.9):
    shape = img.shape
    dim = (shape[1], shape[0])
    heatmap = cv2.resize(heatmap, dim, interpolation = cv2.INTER_AREA)
    # mask = heatmap>0
    # result = (~mask * img) + mask * (alpha * heatmap + (1 - alpha) * img)
    result = img + alpha * heatmap
    return result

def load_alpha_and_proposals(feature_path: str, proposal_path: str, json_path: str, args):
    """ Load alpha and proposals from the given paths. 
    Args:
        feature_path (str): path to the alpha feature
        proposal_path (str): path to the proposals
        json_path (str): path to the json file
        transpose_yz (bool): whether to transpose the y and z axis
    Returns:
        alpha (np.ndarray): alpha feature
        proposals (np.ndarray, Nx6): aabb proposals
        room_bbox (np.ndarray, 6): room bounding box
        res (np.ndarray, 3): resolution of the alpha feature
        boxes_8: (np.ndarray, Nx8x3): 8 corners of the obb proposals in the world coordinate
    """
    feature_npz = np.load(feature_path)
    rgbsigma = feature_npz['rgbsigma']
    res = feature_npz['resolution']
    with open(json_path, 'r') as f:
        json_dict = json.load(f)
        if 'room_bbox' in json_dict:
            room_bbox = np.array(json_dict['room_bbox']).flatten()
        else:
            print('No room_bbox in json file {}'.format(json_path))
    
    # First reshape from (H * W * D, C) to (D, H, W, C)
    # rgbsigma = rgbsigma.reshape(res[2], res[1], res[0], -1)
    alpha = density_to_alpha(rgbsigma[..., -1])

    if args.transpose_yz:
        # Transpose to (D, W, H)
        alpha = np.transpose(alpha, (0, 2, 1))
        res = [res[2], res[0], res[1]]
    else:
        # Transpose to (W, H, D)
        # alpha = np.transpose(alpha, (2, 1, 0))
        res = [res[1], res[2], res[0]]
    
    proposals_npz = np.load(proposal_path)
    if 'proposals' in proposals_npz:
        proposals = proposals_npz['proposals']
    elif 'proposal' in proposals_npz:
        proposals = proposals_npz['proposal']
    else:
        raise ValueError('proposals and proposal are not found in npz.')
    
    proposals = proposals[:args.top_n]
    
    return alpha, proposals, room_bbox, res

def visualize_3dgrid(heatmap, downsample=4):
    heatmap = heatmap[0::downsample, 0::downsample, 0::downsample]
    grid_y, grid_x, grid_z = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), np.arange(heatmap.shape[2]))
    grid = np.concatenate([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1), 
                           heatmap.reshape(-1, 1)], axis=-1)
    grid = np.concatenate([grid, np.ones((grid.shape[0], 1))], axis=-1)
    df = pd.DataFrame(grid, columns=['x', 'y', 'z', 'value', 'species'])
    import plotly.express as px
    fig = px.scatter_3d(df, x='x', y='y', z='z', size='value', color='value', symbol='species')
    fig.show()

def visualize_3dgrid_go(heatmap, downsample=4):
    heatmap = heatmap[0::downsample, 0::downsample, 0::downsample]
    grid_y, grid_x, grid_z = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), np.arange(heatmap.shape[2]))
    x, y, z, heatmap_f = grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), heatmap.flatten()
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=heatmap_f*50,
            color=heatmap_f,                # set color to an array/list of desired values
            colorscale='Hot',   # choose a colorscale
            opacity=0.8,
            symbol='circle'
        )
    )])
    fig.show()

def visualize_volume(heatmap, downsample=4):
    heatmap = heatmap[0::downsample, 0::downsample, 0::downsample]
    grid_y, grid_x, grid_z = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), np.arange(heatmap.shape[2]))
    x, y, z, heatmap_f = grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), heatmap.flatten()
    fig = go.Figure(data=go.Volume(
        x=x,
        y=y,
        z=z,
        value=heatmap_f,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=100, # needs to be a large number for good volume rendering
        ))
    fig.show()

def generate_heatmap(alpha, boxes, args):
    heatmap = np.zeros_like(alpha)
    for box in boxes:
        kernel = np.zeros((box[3]-box[0], box[4]-box[1], box[5]-box[2]))
        if args.kernel_type == 'gaussian':
            kernel = gkern_3d(w=box[3]-box[0], l=box[4]-box[1], h=box[5]-box[2])
        elif args.kernel_type == 'box':
            kernel = np.ones_like(kernel)
        heatmap[box[0]:box[3], box[1]:box[4], box[2]:box[5]] += kernel
    heatmap = gaussian_filter(heatmap, sigma=args.gaussian_sigma)
    mean, std = heatmap.mean(), heatmap.std()
    heatmap = (heatmap - mean) / std
    # print(heatmap.shape) # debug
    return heatmap

def get_overview_position(room_bbox, res, downsample, in_offset=0.3, cam_height=2, index=2):
    x1, y1, x2, y2 = room_bbox[0]+in_offset, room_bbox[1]+in_offset, room_bbox[3]-in_offset, room_bbox[4]-in_offset
    focal_point = np.array([(x1+x2)/2., (y1+y2)/2., 1.0])
    cam_positions = np.array([[x1, y1, cam_height], [x1, y2, cam_height], [x2, y1, cam_height], [x2, y2, cam_height]])
    focal_point = world2grid(focal_point, room_bbox, res, downsample)
    cam_positions = world2grid(cam_positions, room_bbox, res, downsample)

    return focal_point, cam_positions[index]

def frame2config(frame_list, room_bbox, res, downsample=1):
    """ Decode frame list to get camera positions and focal points in grid. 
    """
    names, cam_positions, focal_points, poses = [], [], [], []
    for frame_meta in frame_list:
        name = frame_meta['file_path'].split('/')[-1].split('.')[0]
        names.append(name)

        c2w = np.array(frame_meta['transform_matrix'])
        poses.append(c2w)

        cam_position_world = copy.deepcopy(c2w[:3, 3])
        cam_position = world2grid(cam_position_world, room_bbox, res, downsample)
        cam_positions.append(cam_position)

        focal_point_hom = c2w @ np.array([0, 0, -1, 1]).T
        focal_point_world = focal_point_hom[:3] / focal_point_hom[3]
        focal_point = world2grid(focal_point_world, room_bbox, res, downsample)
        focal_points.append(focal_point)

    
    return names, cam_positions, focal_points, poses

def render_volume(heatmap, room_bbox, res, output_dir, args, json_dict=None, boxes_8=None):
    heatmap = heatmap[0::args.downsample, 0::args.downsample, 0::args.downsample]
    heatmap *= args.value_scale 
    print('mean={}, std={}, min={}, max={}'.format(np.mean(heatmap), np.std(heatmap), np.min(heatmap), np.max(heatmap)))

    # Create the spatial reference
    grid = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
    grid.dimensions = np.array(heatmap.shape) + 1

    # Edit the spatial reference
    grid.origin = (0, 0, 0)  # The bottom left corner of the data set
    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    # Add the data values to the cell data
    grid.cell_data["values"] = heatmap.flatten(order="F")  # Flatten the array!

    # Now plot the grid!
    # grid.plot(show_edges=True)
    # print(grid)

    # cmap choices: plasma, CMRmap, inferno, magma, gist_heat, jet
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    p = pv.Plotter()
    p.add_volume(grid, cmap='jet', opacity='linear', blending='maximum')
    p.background_color = "black"
    # p.add_background_image('/Users/abraham/Desktop/fyp/RPN_NeRF_temp/raw_2.jpg')
    p.window_size = [640, 480]
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.view_angle = 60
    p.remove_scalar_bar()
    
    if json_dict==None: # debug
        focal_point, cam_position = get_overview_position(room_bbox, res, args.downsample, index=2)
        p.camera.position = cam_position
        p.camera.focal_point = focal_point
        p.save_graphic(join(output_dir, "temp/heatmap.svg"))
    else:
        frame_dict = json_dict['frames']
        names, cam_positions, focal_points, poses = frame2config(frame_dict, room_bbox, res, args.downsample)
        fl_x, fl_y, cx, cy = json_dict['fl_x'], json_dict['fl_y'], json_dict['cx'], json_dict['cy']
        intrinsic_mat = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
        for name, cam_position, focal_point, pose in zip(names, cam_positions, focal_points, poses):
            p.camera.position = cam_position
            p.camera.focal_point = focal_point
            if args.interactive:
                p.show()
                break
            else:
                p.save_graphic(join(output_dir, name+'_hmp.svg'))
                renderPM.drawToFile(svg2rlg(join(output_dir, name+'_hmp.svg')), 
                                            join(output_dir, name+'_hmp.png'), fmt='PNG')
                os.remove(join(output_dir, name+'_hmp.svg'))
                if args.concat_img:
                    hmp = cv2.imread(join(output_dir, name+'_hmp.png'))
                    img1 = cv2.imread(join(args.dataset_dir, args.scene_name, 'val/screenshots/'+name+'.jpg'))
                    shape = img1.shape
                    hmp = cv2.resize(hmp, (shape[1], shape[0]), interpolation = cv2.INTER_AREA)
                    img2 = copy.deepcopy(img1)
                    img2 = project_obb_to_image(img2, intrinsic_mat, np.linalg.inv(pose), boxes_8)
                    img = np.concatenate([img1, hmp, img2], axis=1)
                    cv2.imwrite(join(output_dir, name+'_hmp.png'), img)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='path to dataset directory')
    parser.add_argument('--feature_dir', type=str, help='path to feature directory')
    parser.add_argument('--proposal_dir', type=str, help='path to proposal directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--boxes_dir', type=str, help='path to boxes directory')
    parser.add_argument('--transpose_yz', action='store_true', help='transpose y and z')
    parser.add_argument('--top_n', type=int, default=100, help='top n proposals to be used for heatmap.')
    parser.add_argument('--use_gt', action='store_true', help='use ground truth boxes')
    parser.add_argument('--kernel_type', type=str, default='gaussian', choices=['gaussian', 'box'], 
                        help='type of heatmap to be generated')
    parser.add_argument('--value_scale', type=float, default=20, help='value scaling for heatmap')
    parser.add_argument('--downsample', type=int, default=2, help='downsample factor for heatmap')
    parser.add_argument('--gaussian_sigma', type=float, default=5, help='sigma for gaussian kernel')
    parser.add_argument('--concat_img', action='store_true', help='concatenate heatmap with NeRF image')
    parser.add_argument('--interactive', action='store_true', help='interactive mode, used in .ipynb')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    scene_list = [x.split('.')[0] for x in sorted(os.listdir(args.proposal_dir))]
    scene_list = scene_list[:1] # debug

    for scene_name in scene_list:
        args.scene_name = scene_name # pass scene_name to with args
        feature_path = join(args.feature_dir, scene_name+'.npz')
        proposal_path = join(args.proposal_dir, scene_name+'.npz')
        train_json_path = join(args.dataset_dir, scene_name, 'train', 'transforms.json')
        val_json_path = join(args.dataset_dir, scene_name, 'val', 'val_transforms.json')
        assert os.path.isfile(feature_path), 'feature file not found: {}'.format(feature_path)
        assert os.path.isfile(proposal_path), 'proposal file not found: {}'.format(proposal_path)
        assert os.path.isfile(train_json_path), 'train json file not found: {}'.format(train_json_path)
        assert os.path.isfile(val_json_path), 'val json file not found: {}'.format(val_json_path)
        scene_output_dir = join(args.output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        alpha, proposals, room_bbox, res = load_alpha_and_proposals(feature_path, proposal_path, train_json_path, args)
        aabbs = obb2hbb(proposals).astype(int)
        boxes_point8 = grid2world(obb2point8(proposals), room_bbox, res)
        for i in range(3):
            aabbs[:, [i, i+3]] = np.clip(aabbs[:, [i, i+3]], a_min=0, a_max=res[i]-1)
        
        if args.use_gt:
            gt = np.load(join(args.boxes_dir, scene_name+'.npy'))
            aabbs = obb2hbb(gt).astype(int)
            boxes_point8 = grid2world(obb2point8(gt), room_bbox, res)

        heatmap = generate_heatmap(alpha, aabbs, args)

        # render poses in val_json_paths
        with open(val_json_path, 'r') as f:
            val_json_dict = json.load(f)
        
        render_volume(heatmap, room_bbox, res, scene_output_dir, args, val_json_dict, boxes_point8)

    print('Done.')
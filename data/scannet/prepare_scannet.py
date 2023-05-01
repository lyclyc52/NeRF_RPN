import os
import cv2
import json
import shutil
import pandas
import random
import argparse
import subprocess
import numpy as np
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from run_colmap import run_colmap


def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	return fm


def validate_pose(img_idxs, pose_dir):
    valid_idxs = []
    for idx in img_idxs:
        pose_file = os.path.join(pose_dir, f'{idx}.txt')
        pose = np.loadtxt(pose_file)
        if np.isnan(pose).any() or np.isinf(pose).any():
            continue

        valid_idxs.append(idx)

    return valid_idxs


def prepare_scannet_scene(scene_path, output_path, num_train_samples=100, num_val_samples=20):
    scene_name = os.path.basename(scene_path)
    output_path = os.path.join(output_path, scene_name)
    os.makedirs(output_path, exist_ok=True)

    img_dir = os.path.join(scene_path, 'extract', 'color')
    intrinsic_dir = os.path.join(scene_path, 'extract', 'intrinsic')
    pose_dir = os.path.join(scene_path, 'extract', 'pose')

    assert os.path.exists(img_dir)
    assert os.path.exists(intrinsic_dir)
    assert os.path.exists(pose_dir)

    img_idxs = os.listdir(img_dir)
    img_idxs = [x for x in img_idxs if x.endswith('.jpg')]
    img_idxs = [int(x.split('.')[0]) for x in img_idxs]
    img_idxs = sorted(img_idxs)

    img_idxs = validate_pose(img_idxs, pose_dir)

    if len(img_idxs) < 2000:
        print(f'Warning: {scene_path} has less than 2000 valid images')
        return

    img2sharpness = {}

    sampled_idxs = []
    interval = len(img_idxs) // num_train_samples
    for i in range(num_train_samples):
        idxs = img_idxs[i*interval:(i+1)*interval]
        names = [os.path.join(img_dir, f'{x}.jpg') for x in idxs]
        sharpnesses = [sharpness(x) for x in names]

        for j in range(len(sharpnesses)):
            img2sharpness[idxs[j]] = sharpnesses[j]

        max_idx = idxs[np.argmax(sharpnesses)]
        sampled_idxs.append(max_idx)

    sampled_val_idxs = []
    val_interval = len(img_idxs) // num_val_samples
    for i in range(num_val_samples):
        idxs = img_idxs[i*val_interval:(i+1)*val_interval]
        idxs = [x for x in idxs if x not in sampled_idxs]

        if len(idxs) == 0:
            continue

        sharpnesses = [img2sharpness[x] for x in idxs]
        sampled_val_idxs.append(idxs[np.argmax(sharpnesses)])

    print(f'{scene_name}: {len(sampled_idxs)} train samples, {len(sampled_val_idxs)} val samples')

    train_df = pandas.DataFrame([str(i) + '.jpg' for i in sampled_idxs], columns=['image'])
    val_df = pandas.DataFrame([str(i) + '.jpg' for i in sampled_val_idxs], columns=['image'])

    train_df.to_csv(os.path.join(output_path, 'train_set.csv'), index=False, header=False)
    val_df.to_csv(os.path.join(output_path, 'test_set.csv'), index=False, header=False)

    config = {
        'name': scene_name,
        'max_depth': 15.0,
        'dist2m': 1.0,
        'rgb_only': True,
    }

    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    os.makedirs(os.path.join(output_path, 'colmap'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'colmap', 'sparse'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'colmap', 'sparse_train'), exist_ok=True)


def prepare_scenes(scannet_path, output_path, num_scenes, num_train_samples, num_val_samples):
    os.makedirs(output_path, exist_ok=True)

    scene_paths = os.listdir(scannet_path)
    scene_paths = [x for x in scene_paths if os.path.isdir(os.path.join(scannet_path, x))]
    scene_ids = [x.split('_')[0][-4:] for x in scene_paths]
    scene_ids = list(set(scene_ids))
    scene_paths = [os.path.join(scannet_path, f'scene{x}_00') for x in scene_ids]

    valid_scenes = []

    for scene_path in scene_paths:
        img_dir = os.path.join(scene_path, 'extract', 'color')
        intrinsic_dir = os.path.join(scene_path, 'extract', 'intrinsic')
        pose_dir = os.path.join(scene_path, 'extract', 'pose')

        if not os.path.exists(img_dir) or not os.path.exists(intrinsic_dir) or not os.path.exists(pose_dir):
            # print(f'Warning: {scene_path} is not complete')
            continue

        imgs = os.listdir(img_dir)
        if len(imgs) < 2000:
            continue

        valid_scenes.append(scene_path)

    valid_scenes = random.sample(valid_scenes, num_scenes)
    fn = partial(prepare_scannet_scene, output_path=output_path, num_train_samples=num_train_samples, 
                 num_val_samples=num_val_samples)

    process_map(fn, valid_scenes, max_workers=32)


def copy_images(output_path, scene_path):
    os.makedirs(output_path, exist_ok=True)

    img_all_dir = os.path.join(output_path, 'images_all')
    img_train_dir = os.path.join(output_path, 'images_train')

    os.makedirs(img_all_dir, exist_ok=True)
    os.makedirs(img_train_dir, exist_ok=True)

    img_train_src_dir = os.path.join(scene_path, 'train', 'rgb')
    img_test_src_dir = os.path.join(scene_path, 'test', 'rgb')

    img_train = os.listdir(img_train_src_dir)
    img_test = os.listdir(img_test_src_dir)

    for img in img_train:
        shutil.copyfile(os.path.join(img_train_src_dir, img), os.path.join(img_all_dir, img))
        shutil.copyfile(os.path.join(img_train_src_dir, img), os.path.join(img_train_dir, img))

    for img in img_test:
        shutil.copyfile(os.path.join(img_test_src_dir, img), os.path.join(img_all_dir, img))


def extract_scene_rgb(script_path, scannet_path, scene_path):
    cmd = f'{script_path} {scene_path} {scannet_path}'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.communicate()


def copy_colmap_recon(recon_dir, scene_dir):
    sparse_dir = os.path.join(recon_dir, 'sparse')
    sparse_train_dir = os.path.join(recon_dir, 'sparse_train')

    shutil.copytree(sparse_dir, os.path.join(scene_dir, 'colmap', 'sparse'), dirs_exist_ok=True)
    shutil.copytree(sparse_train_dir, os.path.join(scene_dir, 'colmap', 'sparse_train'), dirs_exist_ok=True)


def parse_gpu_ids(gpu_str):
    gpu_ids = []
    if gpu_str:
        for token in gpu_str.split(','):
            if '-' in token:
                start, end = token.split('-')
                gpu_ids.extend(range(int(start), int(end)+1))
            else:
                gpu_ids.append(int(token))
    
    return gpu_ids if gpu_ids else None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare ScanNet dataset for Dense Depth Priors NeRF training.')

    parser.add_argument('--scannet_dir', type=str, help='ScanNet dataset root.')
    parser.add_argument('--output_dir', type=str, help='Output directory.')
    parser.add_argument('--ddp_nerf_dir', type=str, help='Dense Depth Priors NeRF repo root.')

    parser.add_argument('--num_scenes', type=int, default=120, help='Number of scenes to process.')
    parser.add_argument('--num_train_samples', type=int, default=100, help='Number of training samples per scene.')
    parser.add_argument('--num_val_samples', type=int, default=400, help='Number of validation samples per scene.')

    parser.add_argument('--gpu', type=str, default='', help='GPU to use for COLMAP reconstruction.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    gpu_list = parse_gpu_ids(args.gpu)

    scans_dir = os.path.join(args.scannet_dir, 'scans')
    nerf_dir = os.path.join(args.output_dir, 'scannet_nerf')
    colmap_recon_dir = os.path.join(args.output_dir, 'colmap_recon')

    # Select training and validation views
    prepare_scenes(scans_dir, nerf_dir, args.num_scenes, args.num_train_samples, args.num_val_samples)

    # Extract and copy RGB images
    scenes = os.listdir(nerf_dir)
    for scene in tqdm(scenes):
        extract_scene_rgb(os.path.join(args.ddp_nerf_dir, 'preprocessing/build/extract_scannet_scene'),
                          args.scannet_dir, os.path.join(nerf_dir, scene))

        if len(os.listdir(os.path.join(nerf_dir, scene))) == 0:
            print(f'{scene} is empty, removing')
            shutil.rmtree(os.path.join(nerf_dir, scene))
            continue 

        output_dir = os.path.join(colmap_recon_dir, scene)
        copy_images(output_dir, os.path.join(nerf_dir, scene))

    # Run COLMAP reconstruction on all selected views
    scenes = os.listdir(colmap_recon_dir)
    for scene in tqdm(scenes):
        run_colmap(os.path.join(scans_dir, scene), 
                   os.path.join(colmap_recon_dir, scene),
                   './y_down_to_z_up.txt',
                   gpu_list=gpu_list)

    # Copy COLMAP reconstruction to NeRF directory for training
    for scene in tqdm(scenes):
        copy_colmap_recon(os.path.join(colmap_recon_dir, scene, 'recon'),
                          os.path.join(nerf_dir, scene))

        json_path = os.path.join(nerf_dir, scene, 'config.json')

        with open(json_path, 'r') as f:
            data = json.load(f)

        data['rgb_only'] = False

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

#
# Run COLMAP reconstruction for Dense Depth Priors NeRF training views
# Adapted from https://github.com/barbararoessle/dense_depth_priors_nerf/issues/10#issuecomment-1140504572
#

import os
import shutil
import subprocess

import numpy as np
import sqlite3
import quaternion


def list_missing_rgb(rgb_dir, sparse_dir):
    expected_files = os.listdir(rgb_dir)
    found_files = []
    for line in open(os.path.join(sparse_dir, "images.txt")):
        for f in expected_files:
            if " " + f in line:
                found_files.append(f)
                break
    print("Missing: ")
    for exp_f in expected_files:
        if exp_f not in found_files:
            print(exp_f)


# Use ground truth poses for colmap sparse reconstruction
def write_ground_truth_poses(sparse_dir, db_path, pose_dir, camera_params_path):
    print("Writing ground truth poses")

    # Read camera parameters
    with open(camera_params_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.split()[0] == 'fx_color':
                fx = float(line.split()[2])

    # Create files
    with open(os.path.join(sparse_dir, 'points3D.txt'), 'w') as f:
        pass

    # Use colmap estimated camera parameters
    db = sqlite3.connect(db_path)
    db_cursor = db.cursor()

    rows = list(db.execute("SELECT * FROM cameras"))
    assert len(rows) == 1
    camera_id, model, width, height, params, prior = rows[0]
    params = np.frombuffer(params, dtype=np.float64).reshape(-1)

    assert model == 0
    assert params.shape == (3,)

    params = params.copy()
    params[0] = fx / 2      # /2 because of half resolution

    # Remove camera parameters and write new ones
    db.execute("DELETE FROM cameras")
    db.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
        (camera_id, model, width, height, params.tostring(), prior))

    with open(os.path.join(sparse_dir, 'cameras.txt'), 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: 1\n')
        f.write(f'{camera_id} SIMPLE_PINHOLE {width} {height} {params[0]} {params[1]} {params[2]}\n')

    images = db_cursor.execute("SELECT name, image_id FROM images")
    id2name = {img_id: name for name, img_id in images}
    with open(os.path.join(sparse_dir, 'images.txt'), 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')

        for img_id, name in id2name.items():
            pose = np.loadtxt(os.path.join(pose_dir, name.split('.')[0] + ".txt"))
            rot = pose[:3, :3].T
            trans = -rot @ pose[:3, 3]
            quat = quaternion.from_rotation_matrix(rot)
            f.write(f'{img_id} {quat.w} {quat.x} {quat.y} {quat.z} {trans[0]} {trans[1]} {trans[2]} 1 {name}\n\n')

    # convert_cmd = f"colmap model_converter --input_path={sparse_dir} --output_path={sparse_dir} --output_type=BIN"
    # process = subprocess.Popen(convert_cmd, shell=True, stdout=subprocess.PIPE)
    # process.communicate()

    print('Finished writing gt poses, copy images.txt to gt_poses.txt')
    shutil.copyfile(os.path.join(sparse_dir, 'images.txt'), os.path.join(sparse_dir, 'gt_poses.txt'))


def run_colmap_sfm(data_dir, pose_dir, camera_params_path, verbose=False, gpu_list=None):
    print("Running colmap sfm on all images")
    rgb_all_dir = os.path.join(data_dir, "images_all")
    rgb_train_dir = os.path.join(data_dir, "images_train")

    # delete previous failed reconstruction
    recon_dir = os.path.join(data_dir, "recon")
    if os.path.exists(recon_dir):
        shutil.rmtree(recon_dir)

    # run colmap with all images creating database db_all.db
    db_all = os.path.join(recon_dir, "db_all.db")
    sparse_dir = os.path.join(recon_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    extract_cmd = f"colmap feature_extractor  --database_path {db_all} --image_path {rgb_all_dir} " \
                   "--ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE"

    match_cmd = "colmap exhaustive_matcher --database_path {}  --SiftMatching.guided_matching 1".format(db_all)

    if gpu_list:
        gpu_str = ','.join([str(gpu) for gpu in gpu_list])
        extract_cmd += f" --SiftExtraction.gpu_index={gpu_str}"
        match_cmd += f" --SiftMatching.gpu_index={gpu_str}"

    for cmd in [extract_cmd, match_cmd]:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.communicate()

    sparse_dir = os.path.join(sparse_dir, "0")
    os.makedirs(sparse_dir, exist_ok=True)
    write_ground_truth_poses(sparse_dir, db_all, pose_dir, camera_params_path)

    # mapper_cmd = "colmap mapper --database_path {} --image_path {} --output_path {} --Mapper.multiple_model 0".format(db_all, rgb_all_dir, sparse_dir)
    mapper_cmd = f"colmap mapper --database_path {db_all} --image_path {rgb_all_dir} --input_path {sparse_dir} " \
                 f"--output_path {sparse_dir} --Mapper.multiple_model 0"

    triangulate_cmd = f'colmap point_triangulator --database_path {db_all} ' \
                      f'--image_path {rgb_all_dir} --input_path {sparse_dir} ' \
                      f'--output_path {sparse_dir}'

    convert_cmd = "colmap model_converter --input_path={} --output_path={} --output_type=TXT".format(sparse_dir, sparse_dir)
    # colmap_cmds = [extract_cmd, match_cmd, mapper_cmd, convert_cmd]
    colmap_cmds = [triangulate_cmd, mapper_cmd, convert_cmd]

    number_input_images = len(os.listdir(rgb_all_dir))

    for cmd in colmap_cmds:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        for line in process.stdout:
            if verbose:
                print(line)
        process.wait()

    # check completeness of reconstruction
    number_lines = sum(1 for line in open(os.path.join(sparse_dir, "images.txt")))
    number_reconstructed_images = (number_lines - 4) // 2 # 4 lines of comments, 2 lines per reconstructed image
    print("Expect {} images in the reconstruction, got {}".format(number_input_images, number_reconstructed_images))
    if number_input_images == number_reconstructed_images:
        return True
    else:
        list_missing_rgb(rgb_all_dir, sparse_dir)
        return False


def process(data_dir, transform_path, verbose=False, gpu_list=None):

    # transform the reconstruction such that z-axis points up
    print("Transforming reconstruction such that z-axis points up")

    recon_dir = os.path.join(data_dir, "recon")
    rgb_all_dir = os.path.join(data_dir, "images_all")
    rgb_train_dir = os.path.join(data_dir, "images_train")

    sparse_dir = os.path.join(recon_dir, "sparse", "0")
    in_sparse_dir = sparse_dir
    # out_sparse_dir = os.path.join(recon_dir, "sparse{}".format("_y_down"), "0")
    # os.makedirs(out_sparse_dir, exist_ok=True)
    # align_cmd = "colmap model_orientation_aligner --input_path={} --output_path={} --image_path={} --max_image_size={}".format(in_sparse_dir, out_sparse_dir, rgb_all_dir, 640)

    # in_sparse_dir = out_sparse_dir
    out_sparse_dir = os.path.join(recon_dir, "sparse{}".format("_z_up"), "0")
    os.makedirs(out_sparse_dir, exist_ok=True)

    trafo_cmd = f"colmap model_transformer --input_path={in_sparse_dir} --output_path={out_sparse_dir} " \
                f"--transform_path={transform_path}"

    convert_cmd = "colmap model_converter --input_path={} --output_path={} --output_type=TXT".format(out_sparse_dir, out_sparse_dir)

    # colmap_cmds = [align_cmd, trafo_cmd, convert_cmd]
    colmap_cmds = [trafo_cmd, convert_cmd]
    for cmd in colmap_cmds:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        for line in process.stdout:
            if verbose:
                print(line)
        process.wait()

    print('Processing train images')

    # extract features of train images into database db.db
    db = os.path.join(recon_dir, "db.db")
    extract_cmd = f"colmap feature_extractor  --database_path {db} --image_path {rgb_train_dir} " \
                  "--ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE"

    if gpu_list:
        gpu_str = ','.join([str(gpu) for gpu in gpu_list])
        extract_cmd += f" --SiftExtraction.gpu_index={gpu_str}"

    process = subprocess.Popen(extract_cmd, shell=True, stdout=subprocess.PIPE)
    for line in process.stdout:
        if verbose:
            print(line)
    process.wait()

    # copy sparse reconstruction from all images
    constructed_sparse_train_dir = os.path.join(recon_dir, "constructed_sparse_train", "0")
    os.makedirs(constructed_sparse_train_dir, exist_ok=True)
    camera_txt = os.path.join(constructed_sparse_train_dir, "cameras.txt")
    images_txt = os.path.join(constructed_sparse_train_dir, "images.txt")
    points3D_txt = os.path.join(constructed_sparse_train_dir, "points3D.txt")
    shutil.copyfile(os.path.join(out_sparse_dir, "cameras.txt"), camera_txt)
    open(images_txt, 'a').close()
    open(points3D_txt, 'a').close()

    # keep poses of the train images in images.txt and adapt their id to match the id in database db.db
    train_files = os.listdir(rgb_train_dir)
    db_cursor = sqlite3.connect(db).cursor()
    name2dbid = dict((n, id)  for n, id in db_cursor.execute("SELECT name, image_id FROM images"))
    with open(os.path.join(out_sparse_dir, "images.txt"), 'r') as in_f:
        in_lines = in_f.readlines()
    for line in in_lines:
        split_line = line.split(" ")
        line_to_write = None
        if "#" in split_line[0]:
            line_to_write = line
        else:
            for train_file in train_files:
                if " " + train_file in line:
                    db_id = name2dbid[train_file]
                    split_line[0] = str(db_id)
                    line_to_write = " ".join(split_line) + "\n"
                    break
        if line_to_write is not None:
            with open(images_txt, 'a') as out_f:
                out_f.write(line_to_write)

    print('Running exaustive matcher and point triangulator on the train images')

    # run exaustive matcher and point triangulator on the train images
    match_cmd = "colmap exhaustive_matcher --database_path {}  --SiftMatching.guided_matching 1".format(db)

    if gpu_list:
        gpu_str = ','.join([str(gpu) for gpu in gpu_list])
        match_cmd += f" --SiftMatching.gpu_index={gpu_str}"

    sparse_train_dir = os.path.join(recon_dir, "sparse_train", "0")
    os.makedirs(sparse_train_dir, exist_ok=True)

    triangulate_cmd = f"colmap point_triangulator --database_path {db} --image_path {rgb_train_dir} " \
                      f"--input_path {constructed_sparse_train_dir} --output_path {sparse_train_dir}"

    convert_cmd = "colmap model_converter --input_path={} --output_path={} --output_type=TXT".format(sparse_train_dir, sparse_train_dir)

    colmap_cmds = [match_cmd, triangulate_cmd, convert_cmd]
    for cmd in colmap_cmds:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        for line in process.stdout:
            if verbose:
                print(line)
        process.wait()


def run_colmap(scene_dir, output_dir, transform_path, verbose=False, gpu_list=None):
    scene_name = os.path.basename(scene_dir)
    pose_dir = os.path.join(scene_dir, "extract", "pose")
    cam_params_path = os.path.join(scene_dir, f'{scene_name}.txt')

    # run colmap
    succeded = False
    retry_cnt = 0
    while not succeded and retry_cnt < 3:
        succeded = run_colmap_sfm(output_dir, pose_dir, cam_params_path, verbose, gpu_list)
        retry_cnt += 1

    if not succeded:
        raise RuntimeError("Colmap failed")

    # processing
    process(output_dir, transform_path, verbose, gpu_list)

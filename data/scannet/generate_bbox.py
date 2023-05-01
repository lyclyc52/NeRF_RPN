import os
import json
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from tqdm.contrib.concurrent import process_map
from MinimumBoundingBox import MinimumBoundingBox


def find_minimum_bounding_box(vertices):
    '''
    Find the minimum bounding box of a set of points after projected onto xy plane.
    '''
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    proj = vertices[:, :2]
    box = MinimumBoundingBox(proj)

    center = np.array(box.rectangle_center)
    size = np.array((box.length_parallel, box.length_orthogonal))
    angle = box.unit_vector_angle

    center = np.concatenate([center, [(min_z + max_z) / 2]])
    size = np.concatenate([size, [max_z - min_z]])
    obb = np.concatenate([center, size, [angle]])

    return obb


class Instance(object):
    obj_id = 0
    segments = []
    vertices = []
    label = ''
    vertex_positions = []
    min_pt = []
    max_pt = []
    obb = []

    def __init__(self, obj_id, segments, label):
        self.obj_id = obj_id
        self.segments = segments
        self.label = label

    def add_vertices(self, indices):
        seg_set = set(self.segments)
        self.vertices = []
        for i, seg in enumerate(indices):
            if seg in seg_set:
                self.vertices.append(i)

    def add_vertex_positions(self, vertices):
        self.vertex_positions = []
        for i in self.vertices:
            self.vertex_positions.append(vertices[i])

        self.vertex_positions = np.array(self.vertex_positions)
        self.min_pt = np.min(self.vertex_positions, axis=0)
        self.max_pt = np.max(self.vertex_positions, axis=0)
        self.obb = find_minimum_bounding_box(self.vertex_positions)

    def to_dict(self):
        dict = {}
        dict['obj_id'] = self.obj_id
        dict['label'] = self.label
        dict['min_pt'] = self.min_pt.tolist()
        dict['max_pt'] = self.max_pt.tolist()
        dict['obb'] = self.obb.tolist()
        return dict


def load_aggregation(file_path):
    instances = []
    with open(file_path, 'r') as f:
        aggregation = json.load(f)

    for group in aggregation['segGroups']:
        obj_id = group['objectId']
        segments = group['segments']
        label = group['label']
        instances.append(Instance(obj_id, segments, label))

    return instances, aggregation['segmentsFile']


def load_segments(file_path, instances):
    with open(file_path, 'r') as f:
        segments = json.load(f)

    indices = segments['segIndices']

    for instance in instances:
        instance.add_vertices(indices)


def load_ply(file_path, instances, align_mat):
    plydata = PlyData.read(file_path)
    num_verts = plydata['vertex'].count
    vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    vertices[:,0] = plydata['vertex'].data['x']
    vertices[:,1] = plydata['vertex'].data['y']
    vertices[:,2] = plydata['vertex'].data['z']

    # align to the origin
    # pos = np.ones((vertices.shape[0], 4))
    # pos[:, :3] = vertices
    # pos = np.matmul(pos, align_mat.T)
    # vertices = pos[:, :3]
    
    for instance in instances:
        instance.add_vertex_positions(vertices)


def process_scene(scene_path, output_path):
    scene_name = os.path.basename(scene_path)
    meta_path = os.path.join(scene_path, f'{scene_name}.txt')
    with open(meta_path, 'r') as f:
        meta = f.readlines()
        meta = [x for x in meta if 'axisAlignment' in x]
        align_mat = meta[0].strip().strip('axisAlignment = ').split(' ')
        align_mat = [float(x) for x in align_mat]
        align_mat = np.array(align_mat).reshape(4, 4)

    aggre_path = os.path.join(scene_path, f'{scene_name}_vh_clean.aggregation.json')

    instances, segments_file = load_aggregation(aggre_path)
    segments_path = segments_file.replace('scannet.', '')

    load_segments(os.path.join(scene_path, segments_path), instances)
    load_ply(os.path.join(scene_path, f'{scene_name}_vh_clean_2.ply'), instances, align_mat)

    json_dict = {
        'scene_name': scene_name,
        'instances': [instance.to_dict() for instance in instances]
    }

    with open(os.path.join(output_path, f'{scene_name}.json'), 'w') as f:
        json.dump(json_dict, f, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    scenes = os.listdir(args.scene_path)
    scenes = [os.path.join(args.scene_path, scene) for scene in scenes]
    output = [args.output_path] * len(scenes)

    process_map(process_scene, scenes, output, max_workers=16)

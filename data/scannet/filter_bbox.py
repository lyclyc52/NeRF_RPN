import json
import os
import numpy as np
import argparse
from tqdm import tqdm


exlcuded_labels = [
    'shower curtain rod', 'paper towel', 'ledge', 'tape', 'paper towel roll', 'light switch', 'rug', 'faucet', 'ceiling light',
    'oven mitt', 'blinds', 'clothing', 'structure', 'clothes hangers', 'clothes', 'vent', 'tray', 'closet wall', 'handrail',
    'bathroom stall', 'kitchen apron', 'thermostat', 'swiffer', 'sign', 'hose', 'whiteboard eraser', 'closet rod', 'toilet paper',
    'loofa', 'windowsill', 'tube', 'shower door', 'broom', 'hair dryer', 'tv stand', 'books', 'bath walls', 'rolled poster',
    'floor', 'clothes hanger', 'fire alarm', 'dustpan', 'stairs', 'bike lock', 'lamp base', 'slippers', 'hanging', 'globe', 
    'doorframe', 'plunger', 'window', 'book', 'sink', 'toilet paper dispenser', 'shower walls', 'stair', 'shower floor',
    'soap dispenser', 'toothbrush', 'banner', 'cup', 'doors', 'power outlet', 'hand towel', 'curtains', 'clock', 'pipes',
    'wall hanging', 'mouse', 'alarm clock', 'bathroom stall door', 'closet doors', 'towel', 'grab bar', 'closet door',
    'shower wall', 'blackboard', 'paper towel dispenser', 'food display', 'mug', 'mat', 'toilet paper holder', 'ceiling',
    'whiteboard', 'bulletin board', 'tissue box', 'mail', 'scale', 'rope', 'music book', 'mirror', 'decoration', 'painting',
    'shower', 'staircase', 'poster', 'pantry walls', 'curtain', 'shower head', 'light', 'smoke detector', 'pipe', 'paper bag',
    'laundry detergent', 'stair rail', 'projector screen', 'cutting board', 'stapler', 'divider', 'mirror doors', 'paper',
    'board', 'hair brush', 'hand sanitzer dispenser', 'controller', 'plate', 'flip flops', 'shoe', 'door', 'soap dish',
    'toilet flush button', 'picture', 'power strip', 'wall'
]


def filter_bbox(feature_path, obj_json_path, npy_output_path, json_output_path, min_size):
    '''
    Filter out bounding boxes that are too small or are excluded by labels.
    '''
    data = np.load(feature_path)
    with open(obj_json_path) as f:
        json_dict = json.load(f)

    obb = [x['obb'] for x in json_dict['instances']]
    obb = np.array(obb)

    max_pt = [x['max_pt'] for x in json_dict['instances']]
    max_pt = np.array(max_pt)

    min_pt = [x['min_pt'] for x in json_dict['instances']]
    min_pt = np.array(min_pt)

    res = data['resolution']
    bbox_min = np.min(min_pt, axis=0)
    bbox_max = np.max(max_pt, axis=0)

    obb[:, 3:6] = obb[:, 3:6] / (bbox_max - bbox_min) * res
    obb[:, :3] = (obb[:, :3] - bbox_min) / (bbox_max - bbox_min) * res

    labels = [x['label'] for x in json_dict['instances']]

    keep = np.ones(len(obb), dtype=bool)
    for i in range(len(obb)):
        if labels[i] in exlcuded_labels:
            keep[i] = False
        elif np.min(obb[i, 3:6]) < min_size:
            keep[i] = False

    obb = obb[keep]

    np.save(npy_output_path, obb)

    instances_filtered = [json_dict['instances'][i] for i in range(len(json_dict['instances'])) if keep[i]]
    json_dict['instances'] = instances_filtered
    with open(json_output_path, 'w') as f:
        json.dump(json_dict, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, required=True)
    parser.add_argument('--obj_json_dir', type=str, required=True)
    parser.add_argument('--npy_output_dir', type=str, required=True)
    parser.add_argument('--json_output_dir', type=str, required=True)
    parser.add_argument('--min_size', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.npy_output_dir, exist_ok=True)
    os.makedirs(args.json_output_dir, exist_ok=True)

    for scene in tqdm(os.listdir(args.feature_dir)):
        scene_name = scene.split('.')[0]
        feature_path = os.path.join(args.feature_dir, f'{scene_name}.npz')
        obj_json_path = os.path.join(args.obj_json_dir, f'{scene_name}.json')
        npy_output_path = os.path.join(args.npy_output_dir, f'{scene_name}.npy')
        json_output_path = os.path.join(args.json_output_dir, f'{scene_name}.json')
        filter_bbox(feature_path, obj_json_path, npy_output_path, json_output_path, args.min_size)

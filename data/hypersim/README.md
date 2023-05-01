# Hypersim NeRF Detection Dataset Creation

`preprocess_boxes.py` contains the code for creating `.npy` files of object bounding box information from Hypersim. Please refer to Hypersim [repo](https://github.com/apple/ml-hypersim) for dataset download, structures, and docs.

The code and docs for training NeRF from Hypersim scenes and subsequent feature extraction are available in our instant-ngp [fork](https://github.com/zymk9/instant-ngp/tree/master/scripts).

Example of use:
```bash
python preprocess_boxes.py \
--format obb \
--dataset_dir path/to/nerf/scenes \
--feature_dir path/to/extracted/nerf/features \
--output_dir path/to/output \
--label_descs ../ml-hypersim/code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv \
--hypersim_path ../hypersim \
--semantics ../ml-hypersim/evermotion_dataset/scenes \
--filter_by_label \
--filter_by_size 
```

Note that further cleaning of the preprocessed object annotations is needed, as what we have performed on our released dataset. The given script uses class labels provided by Hypersim but does not generate any metadata from it. Although you can easily modify the code to produce it and also customize the classes you want to exclude.

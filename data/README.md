# NeRF-RPN Dataset

We release the three main NeRF object detection datasets used in the paper, which are based on [Hypersim](https://github.com/apple/ml-hypersim), [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) & [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), and [ScanNet](http://www.scan-net.org/). 

We use the multi-view RGB images from Hypersim and ScanNet for NeRF training and clean up some of the 3D object annotation to fit it to our task. For 3D-FRONT NeRF dataset, we render object-centric multi-view images based on the provided room layouts and furniture models, using [BlenderProc](https://github.com/DLR-RM/BlenderProc). 

The released datasets contain posed RGB images, NeRF models, radiance and density extracted from NeRFs, as well as object annotations for each scene. We are also actively expanding & refining the datasets and plan to release more scenes with richer annotations in the future.


## Download
[OneDrive link to the dataset.](https://hkustconnect-my.sharepoint.com/:f:/g/personal/bhuai_connect_ust_hk/Ekjf3YC0W9BMsc-jHWXI4xEBy5s_OJBLEbebNVIprd4zMg?e=FgbN9S)

The data is separated into NeRF data and NeRF-RPN training data. You only need the NeRF-RPN data (around 16GB), containing extracted NeRF RGB, density, and bounding box annotations, to reproduce or test our work.


The NeRF training data and models take around 80GB. They are useful for retraining NeRFs, rendering novel views, or re-extracting NeRF features.


## NeRF Data Organization
We use [instant-ngp](https://github.com/NVlabs/instant-ngp) to train the NeRF for Hypersim and 3D-FRONT, and [dense depth priors NeRF](https://github.com/barbararoessle/dense_depth_priors_nerf) for ScanNet. Therefore, the NeRF data organization is largely identical to the general NeRF data format.

For instance, you can find the following structure under `hypersim_nerf_data`

```
hypersim_nerf_data
|- ai_001_001
|  └- train
|     |- images
|     |  └-...
|     |- model.msgpack
|     └- transforms.json       
└-...
```

where `model.msgpack` is the instant-ngp NeRF model, `transforms.json` contains camera parameters and poses, and `images` contains the RGB images. 

For 3D-FRONT dataset, you can find an extra `overview` folder containing floor plan and overview images of the scene with bounding boxes annotated. 

For ScanNet, the data is organized as required in [dense depth priors NeRF](https://github.com/barbararoessle/dense_depth_priors_nerf), and the model is under the `checkpoint` folder with `.tar` extension. Please check the original repo for usage.

Note that the instant-ngp models are trained with **an earlier version of instant-ngp** which we forked [here](https://github.com/zymk9/instant-ngp/tree/10f337f3467b3992e1ad48a0851aeb029d6642a3). They also use the CUTLASS MLP so make sure `TCNN_CUDA_ARCHITECTURES` is set to 61 when compiling instant-ngp. We also plan to update the models with newer versions of instant-ngp.


## NeRF-RPN Data Organization
NeRF-RPN data contain the dataset split, extracted NeRF density and color, and information of axix-aligned bounding boxes (AABB) and oriented bounding boxes (OBB). For example:

```
front3d_rpn_data
|- aabb
|  |- 3dfront_0000_00.npy
|  └-...
|- features
|  |- 3dfront_0000_00.npz
|  └-...
|- obb
|  |- 3dfront_0000_00.npy
|  └-...
└- 3dfront_split.npz
```

`3dfront_split.npz` contains the train/val/test split used in our paper. `aabb` contains the AABB in each scene, with shape `(N, 6)`, and in the form of `(x_min, y_min, z_min, x_max, y_max, z_max)` for each box. `obb` contains the OBB with yaw angle, with shape `(N, 7)`, in the form of `(x, y, z, w, l, h, theta)`, where $theta$ is the yaw angle about the z axis. 

The extracted NeRF rgb and density are in `features`, which can be loaded with numpy and read the `rgbsigma` entry. The `rgbsigma` values have a shape of `(W, L, H, 4)`, where the last dimension stores `(R, G, B, density)`. `W, L, H` corresponds to scene length in x, y, z axis. Most other attributes stored in the `npz` files are only for visualization purpose and are not used in NeRF-RPN.

**Note that the density for Hypersim has been converted to alpha while others are not.** Also note that instant-ngp and dense depth priors NeRF use different activation functions for density.

So far we only provide OBB data for ScanNet, although you can easily calculate AABB from the original ScanNet dataset. We may also include AABB for ScanNet later.

## Dataset Tools
A set of visualization, pre-processing, and utility tools will be released with the code. We will also add docs and tutorials on how to generate data for more scenes in Hypersim, 3D-FRONT, and ScanNet. 

Currently you may check our forked and modified [instant-ngp](https://github.com/zymk9/instant-ngp) and [BlenderProc](https://github.com/hjk0918/BlenderProc) repo for the code if needed. More detailed docs are coming soon.


## Future Updates
We plan to refine the existing scenes with more accessible NeRF model format and better NeRF quality, etc., and release data of more scenes in the future. We may also release additional annotations and scene data such as object class labels, depth maps, 2D segmentations, and point cloud data, especially for the 3D-FRONT NeRF dataset.


## Acknowledgements
We greatly appreciate the source data from [Hypersim](https://github.com/apple/ml-hypersim), [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), and [ScanNet](http://www.scan-net.org/).

We also appreciate the great work from [BlenderProc](https://github.com/DLR-RM/BlenderProc), allowing us to construct realistic synthetic scenes from 3D-FRONT, and [instant-ngp](https://github.com/NVlabs/instant-ngp), for fast NeRF training and sampling.


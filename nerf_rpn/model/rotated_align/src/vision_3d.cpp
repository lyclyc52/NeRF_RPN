// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "cuda_3d/ROIAlignRotated3D_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_rotated_3d_forward",&ROIAlignRotated3D_forward_cuda,"ROIAlignRotated3D_forward");
  m.def("roi_align_rotated_3d_backward",&ROIAlignRotated3D_backward_cuda,"ROIAlignRotated3D_backward");
}

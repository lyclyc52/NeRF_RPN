// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "cuda/ROIAlignRotated_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_align_rotated_forward",&ROIAlignRotated_forward_cuda,"ROIAlignRotated_forward");
  m.def("roi_align_rotated_backward",&ROIAlignRotated_backward_cuda,"ROIAlignRotated_backward");
}

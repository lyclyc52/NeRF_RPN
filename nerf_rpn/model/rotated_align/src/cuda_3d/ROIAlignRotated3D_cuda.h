#pragma once
#include <torch/extension.h>


at::Tensor ROIAlignRotated3D_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_width,
                                 const int pooled_length,
                                 const int pooled_height,
                                 const int sampling_ratio);

at::Tensor ROIAlignRotated3D_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_width,
                                  const int pooled_length,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int width,
                                  const int length,
                                  const int height,
                                  const int sampling_ratio);
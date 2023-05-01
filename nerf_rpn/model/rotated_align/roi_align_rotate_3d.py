# Copyright (c) HUST loop and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import sys
from os.path import dirname
sys.path.append('/disk1/yliugu/NeRF_RPN_private/rpn_network/model/rotated_align')
import rotated_roi_3d as _C


class _ROIAlignRotated3D(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()

        
        # print(type(spatial_scale))
        # print(input.type())
        # print(roi.type())
        output = _C.roi_align_rotated_3d_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], output_size[2], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, w, l, h = ctx.input_shape
        grad_input = _C.roi_align_rotated_3d_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            output_size[2],
            bs,
            ch,
            w,
            l,
            h,
            sampling_ratio,
        )

        return grad_input, None, None, None, None


roi_align_rotated_3d = _ROIAlignRotated3D.apply


class ROIAlignRotated3D(nn.Module):
    def __init__(self, output_size, sampling_ratio):
        super(ROIAlignRotated3D, self).__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois, spatial_scale):
        return roi_align_rotated_3d(
            input, rois, self.output_size, spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

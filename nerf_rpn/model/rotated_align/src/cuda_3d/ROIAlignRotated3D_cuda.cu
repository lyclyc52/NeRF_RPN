#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <torch/extension.h>
// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T trilinear_interpolate(const T* bottom_data,
    const int width, const int length, const int height, 
    T x, T y, T z,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > height || y < -1.0 || y > length || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int z_low = (int) z;
  int y_low = (int) y;
  int x_low = (int) x;
  int z_high;
  int y_high;
  int x_high;

  if (z_low >= height - 1) {
    z_high = z_low = height - 1;
    z = (T) z_low;
  } else {
    z_high = z_low + 1;
  }

  if (y_low >= length - 1) {
    y_high = y_low = length - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T lz = z - z_low;
  T ly = y - y_low;
  T lx = x - x_low;
  T hz = 1. - lz, hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[(x_low * width + y_low) * length + z_high];
  T v2 = bottom_data[(x_high * width + y_low) * length + z_high];
  T v3 = bottom_data[(x_low * width + y_high) * length + z_high];
  T v4 = bottom_data[(x_high * width + y_high) * length+ z_high];
  T v5 = bottom_data[(x_low * width + y_low) * length + z_low];
  T v6 = bottom_data[(x_high * width + y_low) * length + z_low];
  T v7 = bottom_data[(x_low * width + y_high) * length + z_low];
  T v8 = bottom_data[(x_high * width + y_high) * length+ z_low];

  T w1 = lz * hy * hx, w2 = lz * hy * lx, w3 = lz * ly * hx, w4 = lz * ly * lx;
  T w5 = hz * hy * hx, w6 = hz * hy * lx, w7 = hz * ly * hx, w8 = hz * ly * lx;
  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 * v8);

  return val;
}

template <typename T>
__global__ void RoIAlignRotated3DForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels,
    const int width, const int length, const int height, 
    const int pooled_width, const int pooled_length, const int pooled_height, 
    const int sampling_ratio,
    const T* bottom_rois, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    
    // (n, c, pw, pl, ph) is an element in the pooled output
    int ph = index % pooled_height;
    int pl = (index / pooled_height) % pooled_length;
    int pw = (index / pooled_height / pooled_length) % pooled_width;
    int c = (index / pooled_height / pooled_length / pooled_width) % channels;
    int n = index / pooled_height / pooled_length / pooled_width / channels;

    const T* offset_bottom_rois = bottom_rois + n * 8;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_center_w = offset_bottom_rois[1] * spatial_scale;
    T roi_center_l = offset_bottom_rois[2] * spatial_scale;
    T roi_center_h = offset_bottom_rois[3] * spatial_scale;
    T roi_width = offset_bottom_rois[4] * spatial_scale;
    T roi_length = offset_bottom_rois[5] * spatial_scale;
    T roi_height = offset_bottom_rois[6] * spatial_scale;
    T theta = offset_bottom_rois[7] * M_PI / 180.0;
    

    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (T)1.);
    roi_length = max(roi_length, (T)1.);
    roi_height = max(roi_height, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_l = static_cast<T>(roi_length) / static_cast<T>(pooled_length);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * length * width ;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_l = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_length / pooled_length);
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    T roi_start_h = -roi_height / 2.0;
    T roi_start_l = -roi_length / 2.0;
    T roi_start_w = -roi_width / 2.0;
    T cosTheta = cos(theta);
    T sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_l * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    for (int iz = 0; iz < roi_bin_grid_h; iz ++) // e.g., iy = 0, 1
    {
      const T zz = roi_start_h + ph * bin_size_h + static_cast<T>(iz + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
      for (int iy = 0; iy < roi_bin_grid_l; iy ++) // e.g., iy = 0, 1
      {
        

        const T yy = roi_start_l + pl * bin_size_l + static_cast<T>(iy + .5f) * bin_size_l / static_cast<T>(roi_bin_grid_l); // e.g., 0.5, 1.5
        
        for (int ix = 0; ix < roi_bin_grid_w; ix ++)
        {
          const T xx = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
          
          // Rotate by theta around the center and translate
          T x = xx * cosTheta + yy * sinTheta + roi_center_w;
          T y = yy * cosTheta - xx * sinTheta + roi_center_l;
          T z = zz + roi_center_h;
          // if(index == 0)
          // if(index == 0)
          //   printf("%d %d %d \n", roi_bin_grid_h, roi_bin_grid_l, roi_bin_grid_w);
          //   printf("%d \n"
          T val = trilinear_interpolate(offset_bottom_data, width, length, height, x, y, z, index);
          output_val += val;
        }
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}


template <typename T>
__device__ void trilinear_interpolate_gradient(
    const int width, const int length, const int height, 
    T x, T y, T z,
    T & w1, T & w2, T & w3, T & w4, T & w5, T & w6, T & w7, T & w8,
    int & x_low, int & x_high, int & y_low, int & y_high, int & z_low, int & z_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > height || y < -1.0 || y > length || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
    x_low = x_high = y_low = y_high = z_low = z_high = -1;
    return;
  }
  if (z <= 0) z = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  z_low = (int) z;
  y_low = (int) y;
  x_low = (int) x;

  if (z_low >= height - 1) {
    z_high = z_low = height - 1;
    z = (T) z_low;
  } else {
    z_high = z_low + 1;
  }

  if (y_low >= length - 1) {
    y_high = y_low = length - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T lz = z - z_low;
  T ly = y - y_low;
  T lx = x - x_low;
  T hz = 1. - lz, hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = lz * hy * hx, w2 = lz * hy * lx, w3 = lz * ly * hx, w4 = lz * ly * lx;
  w5 = hz * hy * hx, w6 = hz * hy * lx, w7 = hz * ly * hx, w8 = hz * ly * lx;
  // printf("%f, %f, %f, %f, %f, %f, %f, %f, %f\n", w1, w2, w3,w4,w5,w6,w7,w8);
  return;
}

template <typename T>
__global__ void RoIAlignRotated3DBackwardFeature(const int nthreads, const T* top_diff,
    const int num_rois, const T spatial_scale,
    const int channels, const int width, const int length, const int height,
    const int pooled_width, const int pooled_length, const int pooled_height, 
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pw, pl, ph) is an element in the pooled output
    int ph = index % pooled_height;
    int pl = (index / pooled_height) % pooled_length;
    int pw = (index / pooled_height / pooled_length) % pooled_width;
    int c = (index / pooled_height / pooled_length / pooled_width) % channels;
    int n = index / pooled_height / pooled_length / pooled_width / channels;

    const T* offset_bottom_rois = bottom_rois + n * 8;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_center_w = offset_bottom_rois[1] * spatial_scale;
    T roi_center_l = offset_bottom_rois[2] * spatial_scale;
    T roi_center_h = offset_bottom_rois[3] * spatial_scale;
    T roi_width = offset_bottom_rois[4] * spatial_scale;
    T roi_length = offset_bottom_rois[5] * spatial_scale;
    T roi_height = offset_bottom_rois[6] * spatial_scale;
    T theta = offset_bottom_rois[7] * M_PI / 180.0;

    // T roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    // T roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    // T roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    // T roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (T)1.);
    roi_length = max(roi_length, (T)1.);
    roi_height = max(roi_height, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_l = static_cast<T>(roi_length) / static_cast<T>(pooled_length);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * length * width ;

    int top_offset = (n * channels + c) * pooled_height * pooled_length * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[(pw * pooled_length + pl) * pooled_height + ph];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_l = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_length / pooled_length);
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    T roi_start_h = -roi_height / 2.0;
    T roi_start_l = -roi_length / 2.0;
    T roi_start_w = -roi_width / 2.0;
    T cosTheta = cos(theta);
    T sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_l * roi_bin_grid_w; // e.g. = 4

    for (int iz = 0; iz < roi_bin_grid_h; iz ++) // e.g., iy = 0, 1
    {
      const T zz = roi_start_h + ph * bin_size_h + static_cast<T>(iz + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
      for (int iy = 0; iy < roi_bin_grid_l; iy ++) // e.g., iy = 0, 1
      {
        const T yy = roi_start_l + pl * bin_size_l + static_cast<T>(iy + .5f) * bin_size_l / static_cast<T>(roi_bin_grid_l); // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix ++)
        {
          const T xx = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
          
          // Rotate by theta around the center and translate
          T x = xx * cosTheta + yy * sinTheta + roi_center_w;
          T y = yy * cosTheta - xx * sinTheta + roi_center_l;
          T z = zz + roi_center_h;
          T w1, w2, w3, w4, w5, w6, w7, w8;
          int x_low, x_high, y_low, y_high, z_low, z_high;

          trilinear_interpolate_gradient(width, length, height, x, y, z,
              w1, w2, w3, w4, w5, w6, w7, w8,
              x_low, x_high, y_low, y_high , z_low, z_high,
              index);

          T g1 = top_diff_this_bin * w1 / count;
          T g2 = top_diff_this_bin * w2 / count;
          T g3 = top_diff_this_bin * w3 / count;
          T g4 = top_diff_this_bin * w4 / count;
          T g5 = top_diff_this_bin * w5 / count;
          T g6 = top_diff_this_bin * w6 / count;
          T g7 = top_diff_this_bin * w7 / count;
          T g8 = top_diff_this_bin * w8 / count;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && z_low >= 0 && z_high >= 0)
          {
            atomicAdd(offset_bottom_diff + z_high + height * (y_low + x_low * length), static_cast<T>(g1));
            atomicAdd(offset_bottom_diff + z_high + height * (y_low + x_high * length), static_cast<T>(g2));
            atomicAdd(offset_bottom_diff + z_high + height * (y_high + x_low * width), static_cast<T>(g3));
            atomicAdd(offset_bottom_diff + z_high + height * (y_high + x_high * width), static_cast<T>(g4));
            atomicAdd(offset_bottom_diff + z_low + height * (y_low + x_low * length), static_cast<T>(g5));
            atomicAdd(offset_bottom_diff + z_low + height * (y_low + x_high * length), static_cast<T>(g6));
            atomicAdd(offset_bottom_diff + z_low + height * (y_high + x_low * width), static_cast<T>(g7));
            atomicAdd(offset_bottom_diff + z_low + height * (y_high + x_high * width), static_cast<T>(g8));
          } // if
        } // ix
      } // iy
    }
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignRotated3DBackward


at::Tensor ROIAlignRotated3D_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_width,
                                 const int pooled_length,
                                 const int pooled_height,
                                 const int sampling_ratio) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto width = input.size(2);
  auto length = input.size(3);
  auto height = input.size(4);

  auto output = at::empty({num_rois, channels, pooled_width, pooled_length, pooled_height}, input.options());
  auto output_size = num_rois * pooled_height * pooled_length * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);


  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIAlignRotated3D_forward", [&] {
    RoIAlignRotated3DForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         spatial_scale,
         channels,
         width,
         length,
         height,
         pooled_width,
         pooled_length,
         pooled_height,
         sampling_ratio,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>());
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
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
                                  const int sampling_ratio) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, width, length, height}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "ROIAlignRotated3D_backward", [&] {
    RoIAlignRotated3DBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         num_rois,
         spatial_scale,
         channels,
         width,
         length,
         height,
         pooled_width,
         pooled_length,
         pooled_height,
         sampling_ratio,
         grad_input.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}
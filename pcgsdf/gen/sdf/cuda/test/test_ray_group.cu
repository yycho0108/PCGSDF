
#define MAX_ITER 128

#include <iostream>
#include <vector>

__global__ void ChildKernel(const int2 res, const int2 roi, const int2 offset,
                            const float dmin, const float* const d_in,
                            float* const d_out) {
  const int x = offset.x + blockIdx.x * blockDim.x + threadIdx.x;
  const int y = offset.y + blockIdx.y * blockDim.y + threadIdx.y;
  d_out[x * res.y + y] += d_in[x * res.y + y] - dmin;
}

__global__ void ParentKernel(const int2 res, const int2 roi,
                             const float* const d_in, float* const d_out) {
  const int x0 = roi.x * (blockIdx.x * blockDim.x + threadIdx.x);
  const int y0 = roi.y * (blockIdx.y * blockDim.y + threadIdx.y);

  const int x1 = min(x0 + roi.x, res.x);
  const int y1 = min(y0 + roi.y, res.y);

  // (technically) iterate a few times and find min distance,
  // but here simulate with min over grid
  float dmin{10000.0F};
  for (int x = x0; x < x1; ++x) {
    for (int y = y0; y < y1; ++y) {
      dmin = min(dmin, d_out[x * res.y + y]);
    }
  }

  // Just for fun, populate d_in with current d_min and allow child kernel to
  // complete the rest.
  for (int x = x0; x < x1; ++x) {
    for (int y = y0; y < y1; ++y) {
      d_out[x * res.y + y] = dmin;
    }
  }

  // Launch child thread on each partition starting at the specified d_min.
  const dim3 block_dims = {2, 2};  // ==roi
  const dim3 grid_dims = {1, 1};   // I guess always one, for us?
  ChildKernel<<<grid_dims, block_dims>>>(res, roi, int2{x0, y0}, dmin, d_in,
                                         d_out);
}

int main() {
  // Configure
  const int2 resolution = {32, 32};
  const int2 roi = {2, 2};
  const dim3 block_dims(resolution.x / roi.x, resolution.y / roi.y);
  const dim3 grid_dims(1, 1);

  const int num_bytes = resolution.x * resolution.y * sizeof(float);

  // Fill Host Vector
  std::vector<float> d_in_h(resolution.x * resolution.y);
  for (int i = 0; i < resolution.x * resolution.y; ++i) {
    d_in_h[i] = i;
  }
  std::vector<float> d_out_h(resolution.x * resolution.y);

  // Allocate device vector
  float *d_in, *d_out;
  cudaMalloc((void**)&d_in, num_bytes);
  cudaMalloc((void**)&d_out, num_bytes);

  // Copy + Launch
  cudaMemcpy(d_in, d_in_h.data(), num_bytes, cudaMemcpyHostToDevice);
  ParentKernel<<<block_dims, grid_dims>>>(resolution, roi, d_in, d_out);
  cudaMemcpy(d_out_h.data(), d_out, num_bytes, cudaMemcpyDeviceToHost);

  // Free
  cudaFree(d_in);
  cudaFree(d_out);

  // Visualize
  for (int i = 0; i < resolution.x; ++i) {
    for (int j = 0; j < resolution.y; ++j) {
      std::cout << (d_in_h[i * resolution.y + j] ==
                    d_out_h[i * resolution.y + j])
                << ' ';
    }
    std::cout << std::endl;
  }
}

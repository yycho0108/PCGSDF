#include "cho/gen/cuda/render_jit.hpp"

#include <string>

#define FMT_DEPRECATED_PERCENT 1
#include <fmt/format.h>
#include <Eigen/Geometry>

#include "cho/gen/cuda/jitify.hpp"

#define BLOCK_SIZE 32

static std::string GenerateProgramSource(const std::string &point,
                                         const std::string &subex,
                                         const std::string &scene) {
  // NOTE(yycho0108): Alternatively, copy verbatim.
#define STRINGIFY(x) #x
  const std::string cutil_math_header =
#include "cho/gen/cuda/cutil_math.cuh"
      ;
#undef STRINGIFY

  // Generate program source
  const std::string program_template = R"(RayMarch
    {}
    __global__ void RayMarchKernel(
        const float3 eye, const float4 eye_q, const int2 res, const float2 fov,
        const int max_iter,
        const float max_depth, const float eps,
        float* const depth_image,
        float3* const point_cloud
        ) {{

        // Determine output index.
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= res.x || y >= res.y) {{
            return;
        }}
        const int index = x * res.y + y;

        // Compute ray based on x,y, fov and resolution.
        const float2 fov_step = fov / make_float2(res.x, res.y);
        const float2 angles = fov / -2.0F + make_float2(x, y) * fov_step;
        const float c = cos(angles.x);
        const float3 ray_local{{c * cos(angles.y), c * sin(angles.y), sin(angles.x)}};
        const float3 ray = rotate(eye_q, ray_local);

        float depth = 0;
        for(int i=0; i < max_iter; ++i){{
            const float3 {} = eye + depth * ray;

            // Evaluate Subexpressions.
            {}

            // Evaluate SDF.
            const float offset = {};
            if (offset < eps) {{
                break;
            }}
            depth += offset;
            if (depth >= max_depth) {{
                depth = max_depth;
                break;
            }}
        }}
        depth_image[index] = depth;
        point_cloud[index] = depth * ray;
    }})";
  const std::string program_source =
      fmt::format(program_template, cutil_math_header, point, subex, scene);
  return program_source;
}

void CreateDepthImageCudaJit(const Eigen::Isometry3f &camera_pose,
                             const Eigen::Vector2i &resolution,
                             const Eigen::Vector2f &field_of_view,
                             const std::string &point, const std::string &subex,
                             const std::string &scene,
                             Eigen::MatrixXf *const depth_image,
                             std::vector<Eigen::Vector3f> *const point_cloud) {
  // Convert arguments.
  const float3 eye{camera_pose.translation().x(), camera_pose.translation().y(),
                   camera_pose.translation().z()};
  const Eigen::Quaternionf q{camera_pose.linear()};
  const float4 eye_q{q.x(), q.y(), q.z(), q.w()};
  const int2 res{resolution.x(), resolution.y()};
  const float2 fov{field_of_view.x(), field_of_view.y()};

  // Generate program.
  const std::string program_source = GenerateProgramSource(point, subex, scene);
  static jitify::JitCache kernel_cache;
  jitify::Program program = kernel_cache.program(program_source, 0);

  // Compute buffer size.
  const int depth_bytes = sizeof(float) * res.x * res.y;
  const int cloud_bytes = sizeof(float) * res.x * res.y * 3;

  // Allocate.
  float *depth_d;
  float *cloud_d;
  cudaMalloc((void **)&depth_d, depth_bytes);
  cudaMalloc((void **)&cloud_d, cloud_bytes);

  // Prep kernel dims.
  const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid((res.x + block.x - 1) / block.x,
                  (res.y + block.y - 1) / block.y);

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    if (call != CUDA_SUCCESS) {                                           \
      const char *str;                                                    \
      cuGetErrorName(call, &str);                                         \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      return;                                                             \
    }                                                                     \
  } while (0)
#endif

  CHECK_CUDA(
      program.kernel("RayMarchKernel")
          .instantiate()
          .configure(grid, block)
          .launch(eye, eye_q, res, fov, 128, 100.0F, 1e-6, depth_d, cloud_d));
#undef CHECK_CUDA

  // Allocate.
  depth_image->resize(res.x, res.y);
  point_cloud->resize(res.x * res.y);

  // Copy.
  cudaMemcpy(reinterpret_cast<float *>(depth_image->data()), depth_d,
             depth_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(reinterpret_cast<float *>(point_cloud->data()), cloud_d,
             cloud_bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Free.
  cudaFree(depth_d);
  cudaFree(cloud_d);
}

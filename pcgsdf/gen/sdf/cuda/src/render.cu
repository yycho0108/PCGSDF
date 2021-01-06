#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <chrono>
#include <iostream>

#include "cho/gen/cuda/cutil_math.cuh"
#include "cho/gen/cuda/render.hpp"
#include "cho/gen/sdf_types.hpp"

#define BLOCK_SIZE 32
#define STACK_SIZE 32
#define USE_SHMEM 1
#define OVERSTEP 0

#define PROFILE_SECTIONS 0

__device__ inline void Push(float* const s, int& i, const float value) {
  s[++i] = value;
}

__device__ inline float Pop(float* const s, int& i) { return s[i--]; }

__device__ inline float SdPoint(const float2 p, const float2 target) {
  return length(target - p);
}

__device__ inline float SdLine(const float2 p, const float2 a, const float2 b) {
  const float2 pa = p - a;
  const float2 u = normalize(b - a);
  return cross(u, pa);
}

__device__ inline float SdCone(const float3 p, const float2 q) {
  const float2 w{length(p.x, p.y), p.z};
  const float2 d{w - q};
  const float lql = length(q);
  return d.y * q.y >= q.x * w.x ? length(w.x, d.y)
                                : (q.x <= w.x && q.y * w.y <= d.x * q.x)
                                      ? length(d.x, w.y)
                                      : w.y * (q.x + lql) < -d.x * q.y
                                            ? -w.y
                                            : (d.y * q.x + q.y * w.x) / lql;
}

__device__ float EvaluateSdf(const SdfDataCompact* const ops,
                             const float* const params, const int n,
                             float* const s, const float3& point) {
  // Save on typing...
  using Op = cho::gen::SdfOpCode;

  float3 p = point;
  int index{-1};
  for (int i = 0; i < n; ++i) {
    // NOTE(ycho): assert against stack overflow.
    assert(index < STACK_SIZE);

    const SdfDataCompact& op = ops[i];
    const float* const param = params + op.param_offset;
    switch (op.code) {
      case Op::SPHERE: {
        const float radius = param[0];
        Push(s, index, length(p) - radius);
        break;
      }
      case Op::BOX: {
        const float rx = param[0];
        const float ry = param[1];
        const float rz = param[2];
        const float3 q = abs(p) - make_float3(rx, ry, rz);
        Push(s, index, length(max(q, 0.0F)) + min(max(q), 0.0F));
        break;
      }
      case Op::CYLINDER: {
        const float radius = param[0];
        const float height = param[1];
        const float2 d{sqrt(p.x * p.x + p.y * p.y) - radius, abs(p.z) - height};
        Push(s, index, min(max(d), 0.0F) + length(max(d, 0.0F)));
        break;
      }
      case Op::PLANE: {
        const float3 normal = make_float3(param[0], param[1], param[2]);
        const float d = param[3];
        Push(s, index, dot(normal, p) + d);
        break;
      }
      case Op::TORUS: {
        const float2 radii = make_float2(param[0], param[1]);
        const float2 q = make_float2(length(p.x, p.y) - radii.x, p.z);
        Push(s, index, length(q) - radii.y);
        break;
      }
      case Op::CONE: {
        Push(s, index, SdCone(p, make_float2(param[0], param[1])));
        break;
      }
      case Op::ROUND: {
        const float d = Pop(s, index);
        const float r = param[0];
        Push(s, index, d - r);
        break;
      }
      case Op::NEGATION: {
        const float d = Pop(s, index);
        Push(s, index, -d);
        break;
      }
      case Op::UNION: {
        const float d0 = Pop(s, index);
        const float d1 = Pop(s, index);
        // const float out =
        //    d0 > 0 ? (d1 > 0 ? d0 * d1 / sqrt(d0 * d0 + d1 * d1) : d1)
        //           : (d1 > 0 ? d0 : -sqrt(d0 * d0 + d1 * d1));
        // Push(s, index, out);
        Push(s, index, min(d0, d1));
        break;
      }
      case Op::INTERSECTION: {
        const float d0 = Pop(s, index);
        const float d1 = Pop(s, index);
        Push(s, index, max(d0, d1));
        break;
      }
      case Op::SUBTRACTION: {
        const float d0 = Pop(s, index);
        const float d1 = Pop(s, index);
        Push(s, index, max(d0, -d1));
        break;
      }
      case Op::ONION: {
        const float d0 = Pop(s, index);
        const float thickness = param[0];
        Push(s, index, abs(d0) - thickness);
        break;
      }
      case Op::TRANSLATION: {
        const float tx = param[0];
        const float ty = param[1];
        const float tz = param[2];
        p += make_float3(tx, ty, tz);
        break;
      }
      case Op::ROTATION: {
        const float4 q = make_float4(param[0], param[1], param[2], param[3]);
        p = rotate(q, p);
        break;
      }
      case Op::TRANSFORMATION: {
        const float4 q{param[0], param[1], param[2], param[3]};
        const float3 v{param[4], param[5], param[6]};
        p = rotate(q, p) + v;
        break;
      }
      case Op::SCALE_BEGIN: {
        // modify `point` for the subtree.
        p *= param[0];
        break;
      }
      case Op::SCALE_END: {
        const float d = Pop(s, index);
        Push(s, index, d * param[0]);
        // restore `point` for the suptree.
        p *= param[0];
        break;
      }
    }
  }
  // printf("MAX STACK @ %d\n", max_index);
  // Eqivalent to return Pop(s,index);
  return s[0];
}

inline __device__ float3 ComputeRay(const int2 res, const float2 fov,
                                    const float2 index, const float4 q) {
  const float2 angles = fov * (index / res - 0.5F);
  const float c = cos(angles.x);
  const float3 ray_rel =
      make_float3(c * cos(angles.y), c * sin(angles.y), sin(angles.x));
  return rotate(q, ray_rel);

  // Projection mat
  // const float d = res.x / (2 * tan(fov.x / 2));  // focal distance...
  // const float3 ray_local =
  //    normalize(make_float3(d, x - res.x / 2, y - res.y / 2));
}

__global__ void RayMarchingDepthWithProgramKernel(
    const float3 eye, const float4 eye_q, const int2 res, const float2 fov,
    const SdfDataCompact* const ops, const int num_ops,
    const float* const params, const int num_params, const int max_iter,
    const float max_depth, const float eps, float* const depth_image,
    float3* const point_cloud) {
  // Determine self index and return if OOB.
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= res.x || y >= res.y) {
    return;
  }
  const int index = x * res.y + y;

#if USE_SHMEM
  extern __shared__ uint8_t shmem[];
  SdfDataCompact* const ops_s = reinterpret_cast<SdfDataCompact*>(shmem);
  float* const params_s = reinterpret_cast<float*>(&ops_s[num_ops]);
  const int tid = blockDim.x * threadIdx.x + threadIdx.y;
  // number of threads **within block**, not total.
  const int num_threads = blockDim.x * blockDim.y;
  for (int j = tid; j < num_ops; j += num_threads) {
    ops_s[j] = ops[j];
  }
  for (int j = tid; j < num_params; j += num_threads) {
    params_s[j] = params[j];
  }
  __syncthreads();
#endif

  // Each thread will have its own `stack`.
  // Hopefully it would not overflow...
  float stack[STACK_SIZE];

  // Compute ray based on x,y, fov and resolution.
  const float3 ray = ComputeRay(res, fov, make_float2(x, y), eye_q);

  // Perform RayMarching.

#if 1
  bool hit = false;
#endif
  float depth = 0.0F;

#if OVERSTEP
  float omega = 1.2F;
  float prv_offset = 0.0F;
  float step_length = 0.0F;
#endif

  for (int i = 0; i < max_iter; ++i) {
    const float3 point = eye + depth * ray;
#if USE_SHMEM
    const float offset = EvaluateSdf(ops_s, params_s, num_ops, stack, point);
#else
    const float offset = EvaluateSdf(ops, params, num_ops, stack, point);
#endif

#if OVERSTEP
    const float abs_offset = abs(offset);
    const bool sor_fail = omega > 1 && (abs_offset + prv_offset) < step_length;
    if (sor_fail) {
      step_length -= omega * step_length;
      omega = 1;
    } else {
      step_length = offset * omega;
    }
    prv_offset = abs_offset;
    depth += step_length;
#else
    depth += offset;
#endif
    if (abs(offset) < eps) {
      hit = true;
      break;
    }

    if (depth >= max_depth) {
      depth = max_depth;
      break;
    }
  }

#if 0
  if (!hit) {
    depth = 0;
  }
#endif

  // Output!
  depth_image[index] = depth;
  point_cloud[index] = eye + depth * ray;
}

__device__ float RayMarchSimple(const int max_iter, const float max_depth,
                                const float init_depth,
                                const SdfDataCompact* const ops,
                                const int num_ops, const float* const params,
                                const float3 eye, const float3 ray,
                                const float ray_div, const float eps,
                                float* const stack, int* const hit_iter) {
  float depth = init_depth;
  bool hit = false;

  float prev_offset{0.0F};
  for (int i = 0; i < max_iter; ++i) {
    const float3 point = eye + depth * ray;
    const float offset = EvaluateSdf(ops, params, num_ops, stack, point);

    // Terminate on hit
    if (offset < eps + depth * ray_div) {
      // Optionally restore prior valid point.
      // NOTE(ycho): This is typically triggered on non-zero ray divergence.
      if (ray_div > 0) {
        depth -= prev_offset;
      }

      hit = true;
      *hit_iter = i;
      break;
    }

    depth += offset;
    prev_offset = offset;

    // Terminate past max depth
    if (depth >= max_depth) {
      depth = max_depth;
      break;
    }
  }
  if (!hit) {
    *hit_iter = max_iter;
  }
  return depth;
}

__global__ void RayMarchingDepthWithProgramRayGroupingChildKernel(
    const float3 eye, const float4 eye_q, const int2 res, const float2 fov,
    const int2 offset, const SdfDataCompact* const ops, const int num_ops,
    const float* const params, const int max_iter, const float max_depth,
    const float eps, float* const depth_image, float3* const point_cloud) {
  float stack[STACK_SIZE];
  const int x = offset.x + (blockIdx.x * blockDim.x + threadIdx.x);
  const int y = offset.y + (blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= res.x || y >= res.y) {
    return;
  }
  const float3 ray = ComputeRay(res, fov, make_float2(x, y), eye_q);
  int iter{0};
  const float depth =
      RayMarchSimple(max_iter, max_depth, depth_image[x * res.y + y], ops,
                     num_ops, params, eye, ray, 0.0F, eps, stack, &iter);
  depth_image[x * res.y + y] = depth;
  point_cloud[x * res.y + y] = eye + depth * ray;
}

__global__ void RayMarchingDepthWithProgramRayGroupingKernel(
    const float3 eye, const float4 eye_q, const int2 res, const float2 fov,
    const int2 roi, const SdfDataCompact* const ops, const int num_ops,
    const float* const params, const int num_params, const int max_iter,
    const float max_depth, const float eps, float* const depth_image,
    float3* const point_cloud) {
  // Determine self-index and return if OOB ...
  const int x = roi.x * (blockIdx.x * blockDim.x + threadIdx.x);
  const int y = roi.y * (blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= res.x || y >= res.y) {
    return;
  }

#if 1
  // Copy input to shared memory.
  extern __shared__ uint8_t shmem[];
  SdfDataCompact* const ops_s = reinterpret_cast<SdfDataCompact*>(shmem);
  float* const params_s = reinterpret_cast<float*>(&ops_s[num_ops]);
  const int tid = blockDim.x * threadIdx.x + threadIdx.y;
  const int num_threads = blockDim.x * blockDim.y;
  for (int j = tid; j < num_ops; j += num_threads) {
    ops_s[j] = ops[j];
  }
  for (int j = tid; j < num_params; j += num_threads) {
    params_s[j] = params[j];
  }
  __syncthreads();
#endif

  // Each thread will have its own `stack`.
  // Hopefully it would not overflow...
  float stack[STACK_SIZE];

  // Compute ray based on x,y, fov and resolution.
  const float2 center = make_float2(x, y) + (0.5F * make_float2(roi - 1));
  const float3 ray = ComputeRay(res, fov, center, eye_q);
  const float2 delta_angle = fov * make_float2(roi - 1) / make_float2(2 * res);

  // Scaling factor for ray divergence (radius) as a function of distance
  const float ray_div =
      sqrt(4 - (1 + cos(delta_angle.x)) * (1 + cos(delta_angle.y)));

  // Raymarch along center ray...
  int split_iter{0};
  const float split_depth =
      RayMarchSimple(max_iter, max_depth, 0.0F, ops_s, num_ops, params_s, eye,
                     ray, ray_div, eps, stack, &split_iter);

  // If required, split ray into pieces.
  if (split_depth < max_depth && split_iter < max_iter) {
#if 1
    // Serial
    for (int i = x; i < min(res.x, x + roi.x); ++i) {
      for (int j = y; j < min(res.y, y + roi.y); ++j) {
        const float3 ray = ComputeRay(res, fov, make_float2(i, j), eye_q);
        int iter{0};
        const float depth =
            RayMarchSimple(max_iter - split_iter, max_depth, split_depth, ops,
                           num_ops, params, eye, ray, 0.0F, eps, stack, &iter);
        // const float depth1 =
        //   RayMarchSimple(max_iter, max_depth, 0, ops,
        //                  num_ops, params, eye, ray, 0.0F, eps, stack,
        //                  &iter1);
        // if(depth1 > depth0 + eps){
        //    printf("%f < %f vs truth = %f | %d %d\n", split_depth, depth0,
        //    depth1, iter0, iter1);
        //}
        depth_image[i * res.y + j] = depth;
        point_cloud[i * res.y + j] = eye + depth * ray;
      }
    }
#else
    // Populate initial values
    for (int i = x; i < min(res.x, x + roi.x); ++i) {
      for (int j = y; j < min(res.y, y + roi.y); ++j) {
        depth_image[i * res.y + j] = split_depth;
        point_cloud[i * res.y + j] = eye + split_depth * ray;
      }
    }

    // Parallel
    const dim3 threads(roi.x, roi.y);
    const dim3 blocks(1, 1);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    RayMarchingDepthWithProgramRayGroupingChildKernel<<<blocks, threads, 0,
                                                        s>>>(
        eye, eye_q, res, fov, make_int2(x, y), ops, num_ops, params,
        max_iter - split_iter, max_depth, eps, depth_image, point_cloud);
    cudaStreamDestroy(s);
#endif
  } else {
    // o.w. directly commit result
    // Output!
    for (int i = x; i < min(res.x, x + roi.x); ++i) {
      for (int j = y; j < min(res.y, y + roi.y); ++j) {
        const float3 ray = ComputeRay(res, fov, make_float2(i, j), eye_q);
        depth_image[i * res.y + j] = split_depth;
        point_cloud[i * res.y + j] = eye + split_depth * ray;
      }
    }
  }
}

// Hmm ...
struct SdfDepthImageRendererCu::Impl {
  explicit Impl(const std::vector<cho::gen::SdfData>& scene,
                const Eigen::Vector2i& resolution, const Eigen::Vector2f& fov);

  void SetResolution(const Eigen::Vector2i& resolution);
  void SetFov(const Eigen::Vector2f& fov);

  void Render(const Eigen::Isometry3f& camera_pose,
              Eigen::MatrixXf* const depth_image,
              std::vector<Eigen::Vector3f>* const point_cloud);

 private:
  int2 res;
  float2 fov;

  // Scene Device buffers.
  thrust::device_vector<float> params;
  thrust::device_vector<SdfDataCompact> program;  // compiled sdf

  // Output Device buffers.
  // TODO(yycho0108): Does this need to be dynamic?
  thrust::device_vector<float> depth_image_buf;
  thrust::device_vector<float> point_cloud_buf;
};

// Forwarding calls to `impl`
SdfDepthImageRendererCu::SdfDepthImageRendererCu(
    const std::vector<cho::gen::SdfData>& scene,
    const Eigen::Vector2i& resolution, const Eigen::Vector2f& fov)
    : impl_{std::make_unique<Impl>(scene, resolution, fov)} {}
SdfDepthImageRendererCu::~SdfDepthImageRendererCu() = default;
void SdfDepthImageRendererCu::SetResolution(const Eigen::Vector2i& resolution) {
  impl_->SetResolution(resolution);
}
void SdfDepthImageRendererCu::SetFov(const Eigen::Vector2f& fov) {
  impl_->SetFov(fov);
}

void SdfDepthImageRendererCu::Render(
    const Eigen::Isometry3f& camera_pose, Eigen::MatrixXf* const depth_image,
    std::vector<Eigen::Vector3f>* const point_cloud) {
  impl_->Render(camera_pose, depth_image, point_cloud);
}

// Actual implementation
SdfDepthImageRendererCu::Impl::Impl(const std::vector<cho::gen::SdfData>& scene,
                                    const Eigen::Vector2i& resolution,
                                    const Eigen::Vector2f& fov) {
  // Set intrinsics.
  SetResolution(resolution);
  SetFov(fov);

  // Translate program into compact form.
  int op_index{0};
  int param_index{0};
  thrust::host_vector<SdfDataCompact> program_h;
  thrust::host_vector<float> param_h;
  program_h.resize(scene.size());
  for (const auto& op : scene) {
    // Set program op and store offset to self params.
    program_h[op_index] = SdfDataCompact{op.code, param_index};

    // Copy parameters to contiguous buffer.
    param_h.insert(param_h.end(), op.param.begin(), op.param.end());

    // Increment indices for populating the next op.
    param_index += op.param.size();
    ++op_index;
  }

  // Copy to device memory.
  program = program_h;
  params = param_h;
}

void SdfDepthImageRendererCu::Impl::SetResolution(
    const Eigen::Vector2i& resolution) {
  this->res = int2{resolution.x(), resolution.y()};

  // Modifying resolution triggers allocation of output buffers as well.
  depth_image_buf.resize(resolution.prod());
  point_cloud_buf.resize(resolution.prod() * 3);
}

void SdfDepthImageRendererCu::Impl::SetFov(const Eigen::Vector2f& fov) {
  this->fov = float2{fov.x(), fov.y()};
}

inline __host__ int CeilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

void SdfDepthImageRendererCu::Impl::Render(
    const Eigen::Isometry3f& camera_pose, Eigen::MatrixXf* const depth_image,
    std::vector<Eigen::Vector3f>* const point_cloud) {
  // Convert pose.
  const float3 eye{camera_pose.translation().x(), camera_pose.translation().y(),
                   camera_pose.translation().z()};
  const Eigen::Quaternionf q{camera_pose.linear()};
  const float4 eye_q{q.x(), q.y(), q.z(), q.w()};

#if 1
  // Normal
  // Prep kernel dims.
  const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 blocks((res.x + threads.x - 1) / threads.x,
                    (res.y + threads.y - 1) / threads.y);
  const int shmem_size =
      sizeof(SdfDataCompact) * program.size() + sizeof(float) * params.size();

  // Launch kernel.
  RayMarchingDepthWithProgramKernel<<<blocks, threads, shmem_size>>>(
      eye, eye_q, res, fov, thrust::raw_pointer_cast(program.data()),
      program.size(), thrust::raw_pointer_cast(params.data()), params.size(),
      16, 100.0F, 1e-2, thrust::raw_pointer_cast(depth_image_buf.data()),
      reinterpret_cast<float3*>(
          thrust::raw_pointer_cast(point_cloud_buf.data())));
#else
  // Ray Grouping
  const int2 roi{2, 2};
  const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 blocks(CeilDiv(CeilDiv(res.x, roi.x), threads.x),
                    CeilDiv(CeilDiv(res.y, roi.y), threads.y));
  const int shmem_size =
      sizeof(SdfDataCompact) * program.size() + sizeof(float) * params.size();
  RayMarchingDepthWithProgramRayGroupingKernel<<<blocks, threads, shmem_size>>>(
      eye, eye_q, res, fov, roi, thrust::raw_pointer_cast(program.data()),
      program.size(), thrust::raw_pointer_cast(params.data()), params.size(),
      32, 100.0F, 1e-2, thrust::raw_pointer_cast(depth_image_buf.data()),
      reinterpret_cast<float3*>(
          thrust::raw_pointer_cast(point_cloud_buf.data())));
#endif

  // Export data.
  depth_image->resize(res.x, res.y);
  thrust::copy(depth_image_buf.begin(), depth_image_buf.end(),
               depth_image->data());

  point_cloud->resize(res.x * res.y);
  thrust::copy(point_cloud_buf.begin(), point_cloud_buf.end(),
               reinterpret_cast<float*>(point_cloud->data()));
}

#if 0
// device
__host__ void CreateDepthImageCuda(const Eigen::Isometry3f& camera_pose,
                                   const Eigen::Vector2i& resolution,
                                   const Eigen::Vector2f& field_of_view,
                                   const std::vector<cho::gen::SdfData>& scene,
                                   Eigen::MatrixXf* const depth,
                                   std::vector<Eigen::Vector3f>* const cloud) {
#if PROFILE_SECTIONS
  auto t0 = std::chrono::high_resolution_clock::now();
#endif

  // Determine number of params.
  int num_params{0};
  for (const auto& op : scene) {
    num_params += op.param.size();
  }

  // Translate program into compact form.
  thrust::device_vector<float> params(num_params);
  thrust::device_vector<SdfDataCompact> program(scene.size());
  int op_index{0};
  int param_index{0};
  // float* const ptr = thrust::raw_pointer_cast(params.data());
  for (const auto& op : scene) {
    program[op_index] = SdfDataCompact{op.code, param_index};
    thrust::copy(op.param.begin(), op.param.end(),
                 params.begin() + param_index);

    // Increment indices for populating the next op.
    param_index += op.param.size();
    ++op_index;
  }

  // Convert arguments.
  const float3 eye{camera_pose.translation().x(), camera_pose.translation().y(),
                   camera_pose.translation().z()};
  const Eigen::Quaternionf q{camera_pose.linear()};
  const float4 eye_q{q.x(), q.y(), q.z(), q.w()};
  const int2 res{resolution.x(), resolution.y()};
  const float2 fov{field_of_view.x(), field_of_view.y()};

  // Allocate output.
  thrust::device_vector<float> depth_image_buf(resolution.x() * resolution.y());
  thrust::device_vector<float> point_cloud_buf(resolution.x() * resolution.y() *
                                               3);

  // Prep kernel dims.
  const dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 blocks((resolution.x() + threads.x - 1) / threads.x,
                    (resolution.y() + threads.y - 1) / threads.y);

#if PROFILE_SECTIONS
  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
#endif

  // TODO(yycho0108): Remove hardcoded params `max_iter`, `max_depth`, `eps`.
  const int shmem_size =
      sizeof(SdfDataCompact) * program.size() + sizeof(float) * params.size();
  RayMarchingDepthWithProgramKernel<<<blocks, threads, shmem_size>>>(
      eye, eye_q, res, fov, thrust::raw_pointer_cast(program.data()),
      program.size(), thrust::raw_pointer_cast(params.data()), params.size(),
      16, 100.0, 1e-3, thrust::raw_pointer_cast(depth_image_buf.data()),
      reinterpret_cast<float3*>(
          thrust::raw_pointer_cast(point_cloud_buf.data())));
#if PROFILE_SECTIONS
  // FIXME(yycho0108): Is this required?
  cudaDeviceSynchronize();
  auto t2 = std::chrono::high_resolution_clock::now();
#endif

  // export data.
  depth->resize(resolution.x(), resolution.y());
  thrust::copy(depth_image_buf.begin(), depth_image_buf.end(), depth->data());

  cloud->resize(resolution.prod());
  thrust::copy(point_cloud_buf.begin(), point_cloud_buf.end(),
               reinterpret_cast<float*>(cloud->data()));
#if PROFILE_SECTIONS
  cudaDeviceSynchronize();
  auto t3 = std::chrono::high_resolution_clock::now();
#endif

#if PROFILE_SECTIONS
  printf(
      "PREP %d\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
  printf(
      "KERNEL %d\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
  printf(
      "EXPORT %d\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count());
#endif
}
#else
__host__ void CreateDepthImageCuda(const Eigen::Isometry3f& camera_pose,
                                   const Eigen::Vector2i& resolution,
                                   const Eigen::Vector2f& field_of_view,
                                   const std::vector<cho::gen::SdfData>& scene,
                                   Eigen::MatrixXf* const depth,
                                   std::vector<Eigen::Vector3f>* const cloud) {
  // Instantiate renderer.
  SdfDepthImageRendererCu{scene, resolution, field_of_view}.Render(
      camera_pose, depth, cloud);
}

#endif

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

__device__ __inline__ void Push(float* const s, int& i, const float value) {
  s[++i] = value;
}

__device__ __inline__ float Pop(float* const s, int& i) { return s[i--]; }

__device__ float EvaluateSdf(const SdfDataCompact* const ops,
                             const float* const params, const int n,
                             float* const s, const float3& point) {
  // Avoid typing too much...
  using Op = cho::gen::SdfOpCode;

  float3 p = point;
  int index{-1};
  // int max_index{0};
  for (int i = 0; i < n; ++i) {
    // will break otherwise.
    assert(index < STACK_SIZE);
    // max_index=max(index,max_index);

    const SdfDataCompact& op = ops[i];
    const float* const param = params + op.param_offset;
    switch (op.code) {
      case Op::SPHERE: {
        const float radius = param[0];
        Push(s, index, sqrt(dot(p, p)) - radius);
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
        const float2 d{sqrt(p.x * p.x + p.y * p.y) - radius,
                       std::abs(p.z) - height};
        Push(s, index, min(max(d), 0.0F) + length(max(d, 0.0F)));
        break;
      }
      case Op::PLANE: {
        const float3 normal = make_float3(param[0], param[1], param[2]);
        const float d = param[3];
        Push(s, index, dot(normal, p) + d);
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
        Push(s, index, std::abs(d0) - thickness);
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
  extern __shared__ std::uint8_t shmem[];
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
#if 0
  // Spherical?
  const float2 fov_step = fov / make_float2(res);
  const float2 angles = fov / -2.0F + make_float2(x, y) * fov_step;
  const float c = cos(angles.x);
  const float3 ray_local{c * cos(angles.y), c * sin(angles.y), sin(angles.x)};
#else
  // Projection mat
  const float d = res.x / (2 * tan(fov.x / 2));  // focal distance...
  const float3 ray_local =
      normalize(make_float3(d, x - res.x / 2, y - res.y / 2));
#endif
  const float3 ray = rotate(eye_q, ray_local);

  // Perform RayMarching.
  bool hit = false;
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
    if (std::abs(offset) < eps) {
      hit = true;
      break;
    }

    // if (depth < 0) {
    //  printf("Somehow inside, abort\n");
    //  depth = 0;
    //  break;
    //}

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

void SdfDepthImageRendererCu::Impl::Render(
    const Eigen::Isometry3f& camera_pose, Eigen::MatrixXf* const depth_image,
    std::vector<Eigen::Vector3f>* const point_cloud) {
  // Convert pose.
  const float3 eye{camera_pose.translation().x(), camera_pose.translation().y(),
                   camera_pose.translation().z()};
  const Eigen::Quaternionf q{camera_pose.linear()};
  const float4 eye_q{q.x(), q.y(), q.z(), q.w()};

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
      32, 100.0, 1e-2, thrust::raw_pointer_cast(depth_image_buf.data()),
      reinterpret_cast<float3*>(
          thrust::raw_pointer_cast(point_cloud_buf.data())));

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

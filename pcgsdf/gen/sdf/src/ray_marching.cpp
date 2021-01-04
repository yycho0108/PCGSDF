#include "cho/gen/ray_marching.hpp"

#include <fmt/printf.h>

#include "cho/gen/sdf.hpp"

namespace cho {
namespace gen {
float RayMarchingDepth(const Eigen::Vector3f& eye, const Eigen::Vector3f& ray,
                       const SdfPtr& sdf, const int max_iter,
                       const float max_depth, const float eps) {
  float depth = 0.0F;
  for (int i = 0; i < max_iter; ++i) {
    const Eigen::Vector3f point = eye + depth * ray;
    const float offset = sdf->Distance(point);
    if (offset < eps) {
      return depth;
    }
    depth += offset;
    if (depth >= max_depth) {
      return max_depth;
    }
  }
  return depth;
}

float RayMarchingDepthWithProgram(const Eigen::Vector3f& eye,
                                  const Eigen::Vector3f& ray,
                                  const std::vector<SdfData>& program,
                                  const int max_iter, const float max_depth,
                                  const float eps) {
  float depth = 0.0F;
  for (int i = 0; i < max_iter; ++i) {
    const Eigen::Vector3f point = eye + depth * ray;
    const float offset = EvaluateSdf(program, point);
    if (offset < eps) {
      return depth;
    }
    depth += offset;
    if (depth >= max_depth) {
      return max_depth;
    }
  }
  return depth;
}
}  // namespace gen
}  // namespace cho

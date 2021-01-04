#pragma once

#include <Eigen/Core>

#include "cho/gen/sdf_fwd.hpp"

namespace cho {
namespace gen {

float RayMarchingDepth(
    const Eigen::Vector3f& eye, const Eigen::Vector3f& ray, const SdfPtr& sdf,
    const int max_iter,
    const float max_depth = std::numeric_limits<float>::infinity(),
    const float eps = std::numeric_limits<float>::epsilon());

float RayMarchingDepthWithProgram(
    const Eigen::Vector3f& eye, const Eigen::Vector3f& ray,
    const std::vector<SdfData>& program, const int max_iter,
    const float max_depth = std::numeric_limits<float>::infinity(),
    const float eps = std::numeric_limits<float>::epsilon());

}  // namespace gen
}  // namespace cho

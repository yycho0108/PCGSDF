#pragma once

#include <random>

#include <Eigen/Core>

#include "cho/gen/sdf_fwd.hpp"

namespace cho {
namespace gen {

cho::gen::SdfPtr GenerateObject(std::default_random_engine &rng,
                                const int num_primitives,
                                const cho::gen::SdfPtr &scene,
                                const cho::gen::SdfPtr &scene_vol,
                                const Eigen::Vector3f &eye);

cho::gen::SdfPtr GenerateSpace(std::default_random_engine &rng,
                               const int num_boxes, const Eigen::Vector3f &eye,
                               cho::gen::SdfPtr *const vol);
}  // namespace gen
}  // namespace cho

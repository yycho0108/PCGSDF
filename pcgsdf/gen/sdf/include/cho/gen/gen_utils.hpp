#pragma once

#include <random>

#include <Eigen/Core>

#include "cho/gen/sdf_fwd.hpp"

namespace cho {
namespace gen {

struct SpaceOptions {
  int min_num_boxes;
  int max_num_boxes;

  float min_box_size;
  float max_box_size;
};

struct ObjectOptions {
  int min_num_primitives;
  int max_num_primitives;
};

struct SceneOptions {
  SpaceOptions space_opts;
  ObjectOptions object_opts;
  int min_num_objects;
  int max_num_objects;
};

cho::gen::SdfPtr GenerateObject(std::default_random_engine &rng,
                                const ObjectOptions &opts,
                                const cho::gen::SdfPtr &scene,
                                const cho::gen::SdfPtr &scene_vol,
                                const Eigen::Vector3f &eye);

cho::gen::SdfPtr GenerateSpace(std::default_random_engine &rng,
                               const SpaceOptions &opts,
                               const Eigen::Vector3f &eye,
                               cho::gen::SdfPtr *const vol);

cho::gen::SdfPtr GenerateScene(std::default_random_engine &rng,
                               const Eigen::Vector3f &eye,
                               const SceneOptions &opts,
                               cho::gen::SdfPtr *const objs_out,
                               cho::gen::SdfPtr *const free_space,
                               cho::gen::SdfPtr *const wall);

// cho::gen::SdfPtr GenerateTrajectory();

}  // namespace gen
}  // namespace cho

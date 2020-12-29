
#include <vector>

#include <Eigen/Geometry>

#include "cho/gen/sdf_types.hpp"
#include "cho/gen/cuda/common.hpp"

void CreateDepthImageCuda(const Eigen::Isometry3f &camera_pose,
                          const Eigen::Vector2i &resolution,
                          const Eigen::Vector2f &fov,
                          const std::vector<cho::gen::SdfData> &scene,
                          Eigen::MatrixXf *const depth_image,
                          std::vector<Eigen::Vector3f> *const point_cloud);

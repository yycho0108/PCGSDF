#pragma once

#include <vector>

#include <Eigen/Geometry>

void CreateDepthImageCudaJit(const Eigen::Isometry3f &camera_pose,
                             const Eigen::Vector2i &resolution,
                             const Eigen::Vector2f &fov,
                             const std::string &point, const std::string &subex,
                             const std::string &scene,
                             Eigen::MatrixXf *const depth_image,
                             std::vector<Eigen::Vector3f> *const point_cloud);

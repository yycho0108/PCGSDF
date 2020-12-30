#pragma once

#include <memory>
#include <vector>

#include <Eigen/Geometry>

#include "cho/gen/cuda/common.hpp"
#include "cho/gen/sdf_types.hpp"

struct SdfDepthImageRendererCu {
  explicit SdfDepthImageRendererCu(const std::vector<cho::gen::SdfData>& scene,
                                   const Eigen::Vector2i& resolution,
                                   const Eigen::Vector2f& fov);
  ~SdfDepthImageRendererCu();

  void SetResolution(const Eigen::Vector2i& resolution);
  void SetFov(const Eigen::Vector2f& fov);

  void Render(const Eigen::Isometry3f& camera_pose,
              Eigen::MatrixXf* const depth_image,
              std::vector<Eigen::Vector3f>* const point_cloud);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void CreateDepthImageCuda(const Eigen::Isometry3f& camera_pose,
                          const Eigen::Vector2i& resolution,
                          const Eigen::Vector2f& fov,
                          const std::vector<cho::gen::SdfData>& scene,
                          Eigen::MatrixXf* const depth_image,
                          std::vector<Eigen::Vector3f>* const point_cloud);

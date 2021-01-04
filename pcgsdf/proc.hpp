#pragma once

#include "kdtree.hpp"
#include "types.hpp"

#include <Eigen/Eigen>

namespace cho {

struct SearchOptions {
    int num_neighbors{ 0 };
    float radius_squared{ 0 };
    bool use_radius{ false };
};

void ComputeMatches(const Cloud3f& src, const Cloud3f& dst,
    std::vector<int>* const out,
    const std::optional<KDTree>& dst_tree = std::nullopt);

void ComputeCovariances(
    const Cloud3f& src,
    const Cloud3f& dst,
    const SearchOptions& opts,
    std::vector<Eigen::Matrix3f>* const covs,
    const std::optional<KDTree>& dst_tree = std::nullopt);

void ComputeNormals(
    const Cloud3f& src,
    const Cloud3f& dst,
    const SearchOptions& opts,
    Cloud3f* const normals,
    const std::optional<KDTree>& dst_tree = std::nullopt);
}

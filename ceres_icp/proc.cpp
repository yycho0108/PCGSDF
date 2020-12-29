
#include "proc.hpp"

namespace cho {
void ComputeMatches(const Cloud3f& src, const Cloud3f& dst,
    std::vector<int>* const out,
    const std::optional<KDTree>& tree_in)
{
    const KDTree& tree = tree_in.has_value() ? tree_in.value() : KDTree{ std::cref(src) };
    out->resize(src.cols());
    for (int i = 0; i < src.cols(); ++i) {
        int j;
        float odsq;
        tree.query(src.col(i).data(), 1, &j, &odsq);
        (*out)[i] = j;
    }
}

inline Eigen::Vector3f ComputeNormalFromCov(const Eigen::Matrix3f& cov)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig{ cov };
    return eig.eigenvectors().col(0);
}

namespace detail {
    void ComputeNormalsWithRadius(
        const Cloud3f& src,
        const KDTree& tree,
        const float radius_squared,
        Cloud3f* const normals)
    {
        const auto& dst = tree.data.get();
        std::vector<std::pair<int, float>> nbrs;
        for (int i = 0; i < src.cols(); ++i) {
            // Search neighbors.
            tree.index->radiusSearch(src.col(i).data(), radius_squared, nbrs,
                nanoflann::SearchParams{ 0, 0, false });
            if (nbrs.empty()) {
                normals->col(i).setZero();
                continue;
            }

            // Compute mean.
            Eigen::Vector3f mean = Eigen::Vector3f::Zero();
            for (const auto& id : nbrs) {
                mean += dst.col(id.first);
            }
            mean /= nbrs.size();

            // Compute covariance.
            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
            for (const auto& id : nbrs) {
                const Eigen::Vector3f delta = dst.col(id.first) - mean;
                cov += delta * delta.transpose();
            }

            normals->col(i) = ComputeNormalFromCov(cov);
        }
    }

    void ComputeNormalsWithKnn(
        const Cloud3f& src,
        const KDTree& tree,
        const int num_neighbors,
        Cloud3f* const normals)
    {
        const auto& dst = tree.data.get();
        std::vector<int> nbr_indices(num_neighbors);
        std::vector<float> nbr_distances(num_neighbors);
        for (int i = 0; i < src.cols(); ++i) {
            // Search neighbors.
            tree.index->knnSearch(src.col(i).data(), num_neighbors,
                nbr_indices.data(), nbr_distances.data());

            // Compute mean.
            Eigen::Vector3f mean = Eigen::Vector3f::Zero();
            for (const auto& j : nbr_indices) {
                mean += dst.col(j);
            }
            mean /= nbr_indices.size();

            // Compute covariance.
            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
            for (const auto& j : nbr_indices) {
                const Eigen::Vector3f delta = dst.col(j) - mean;
                cov += delta * delta.transpose();
            }

            normals->col(i) = ComputeNormalFromCov(cov);
        }
    }
}

void ComputeCovariances(
    const Cloud3f& src,
    const Cloud3f& dst,
    const SearchOptions& opts,
    std::vector<Eigen::Matrix3f>* const covs,
    const std::optional<KDTree>& dst_tree = std::nullopt){

}

void ComputeNormals(
    const Cloud3f& src,
    const Cloud3f& dst,
    const SearchOptions& opts,
    Cloud3f* const normals,
    std::optional<KDTree> tree_in)
{
    const KDTree& tree = tree_in.has_value() ? tree_in.value() : KDTree{ std::cref(src) };

    if (opts.use_radius) {
        detail::ComputeNormalsWithRadius(src, tree, opts.radius_squared, normals);
    } else {
        detail::ComputeNormalsWithKnn(src, tree, opts.num_neighbors, normals);
    }
}
}

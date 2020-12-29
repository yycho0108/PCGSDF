#include <algorithm>
#include <iostream>
#include <random>

#include <ceres/problem.h>
#include <ceres/solver.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/se3.hpp>

using Cloud3f = Eigen::MatrixX3f;

class UniformPose2DDistribution {
public:
    explicit UniformPose2DDistribution(const float pos_min, const float pos_max)
        : x_dist(pos_min, pos_max)
        , y_dist(pos_min, pos_max)
        , yaw_dist(-M_PI, M_PI){};

    template <class Generator>
    Eigen::Isometry2f operator()(Generator& g)
    {
        Eigen::Isometry2f out = Eigen::Translation2f{ x_dist(g), y_dist(g) } * Eigen::Rotation2Df{ yaw_dist(g) };
        return out;
    }

private:
    std::uniform_real_distribution<float> x_dist;
    std::uniform_real_distribution<float> y_dist;
    std::uniform_real_distribution<float> yaw_dist;
};

void TestUP2DD()
{
    std::default_random_engine rd;
    UniformPose2DDistribution dist{ -3.0, 3.0 };
    std::vector<Eigen::Isometry2f> poses;
    std::generate_n(std::back_inserter(poses), 128,
        [&rd, &dist]() -> Eigen::Isometry2f { return dist(rd); });
    for (const auto& pose : poses) {
        std::cout << pose.matrix() << std::endl;
    }
}

template <typename Derived>
class DataSourceBase {
public:
    using Ptr = std::shared_ptr<DataSourceBase>;
    explicit DataSourceBase(){};
    ~DataSourceBase() {}
    int GetSize() const { return static_cast<const Derived*>(this)->GetSize(); }
    const Cloud3f& GetCloud(const int index) const
    {
        return static_cast<const Derived*>(this)->GetCloud(index);
    }
};

class MeshSource : public DataSourceBase<MeshSource> {
public:
    using Ptr = std::shared_ptr<MeshSource>;
    explicit MeshSource(const std::string& filename);
};

class RandomSource : public DataSourceBase<RandomSource> {
public:
    using Ptr = std::shared_ptr<RandomSource>;
    explicit RandomSource() {}
    ~RandomSource() {}
    int GetSize() const { return clouds_.size(); }
    const Cloud3f& GetCloud(const int index) const { return clouds_.at(index); }

    template <typename SizeDistribution, typename PointDistribution>
    static Ptr Create(const int length, const SizeDistribution& size_dist,
        const PointDistribution point_dist)
    {
        //// Generate Clouds ...
        // clouds_(length) {
        //    std::generate(clouds_.begin(), clouds_.end(), []() -> Cloud3f {
        //        Cloud3f out;
        //        return out;
        //    });
        //}
    }

private:
    std::vector<Cloud3f> clouds_;
};

class RepeatFilter : public DataSourceBase<RepeatFilter> {
public:
    using Ptr = std::shared_ptr<RepeatFilter>;
    explicit RepeatFilter(const DataSourceBase::Ptr& source, const int size)
        : source_(source)
        , size_(size)
    {
    }

    int GetSize() const { return size_; }
    const Cloud3f& GetCloud(const int index) const
    {
        if (index < 0 || index >= size_) {
            throw std::out_of_range("out of bounds");
        }
        return source_->GetCloud(0);
    }

private:
    DataSourceBase::Ptr source_;
    int size_;
};

/**
 * @brief Add velocity-induced transforms to the input point cloud sequence.
 */
class TrajectoryFilter : public DataSourceBase<TrajectoryFilter> {
public:
    using Ptr = std::shared_ptr<TrajectoryFilter>;
    explicit TrajectoryFilter(const DataSourceBase::Ptr& source,
        const Eigen::Isometry2f& initial_pose) {}

    int GetSize() const { return source_->GetSize(); }
    const Cloud3f& GetCloud(const int index) const
    {
        if (stale_) {
            clouds_.at(index) = poses_.at(index) * source_->GetCloud(index);
        }
        return clouds_.at(index);
    }

    template <typename... Args>
    static Ptr Create(Args... args)
    {
        return std::make_shared<TrajectoryFilter>(args...);
    }

private:
    DataSourceBase::Ptr source_;
    std::vector<Eigen::Isometry3f> poses_;
    mutable std::vector<Cloud3f> clouds_;
    bool stale_{ true };
};

/**
 * @brief Apply raycast based filter to process self-occlusion.
 */
class RaycastFilter : public DataSourceBase<RaycastFilter> {
};

int main() { return 0; }

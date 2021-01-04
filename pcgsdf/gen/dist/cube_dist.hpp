#pragma once

#include <random>

#include <Eigen/Core>

namespace cho {
namespace gen {

    class CubeSurfaceDistribution {
        explicit CubeSurfaceDistribution(const float size)
            : dist_{ -size / 2, size / 2 }
        {
        }
        template <typename Generator>
        Eigen::Vector3f operator()(Generator& g)
        {
        }

    private:
        std::uniform_real_distribution<float> dist_;
    };
}
}

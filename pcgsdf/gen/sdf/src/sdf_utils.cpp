#include "cho/gen/sdf.hpp"

#include "cho/gen/sdf_utils.hpp"

namespace cho {
namespace gen {

float ComputeSdfVolumeMonteCarlo(const cho::gen::SdfPtr &sdf,
                                 const int num_samples) {
  const Eigen::AlignedBox3f aabb{sdf->Center().array() - sdf->Radius(),
                                 sdf->Center().array() + sdf->Radius()};
  int count{0};
  for (int i = 0; i < num_samples; ++i) {
    count += (sdf->Distance(aabb.sample()) > 0);
  }
  return count / num_samples * std::pow(2 * sdf->Radius(), 3);
}

cho::gen::SdfPtr MakeTangentApprox(const cho::gen::SdfPtr &source,
                                   const cho::gen::SdfPtr &target,
                                   const bool force_tangent) {
  const Eigen::Vector3f delta = target->Center() - source->Center();
  const float d1 = source->Distance(target->Center());
  const float d2 = target->Distance(source->Center());
  const float offset = (d1 + d2 - delta.norm());
  // Don't `maketangent` if already intersecting
  if (force_tangent || offset > 0) {
    return cho::gen::OpTransformation::Create(
        source, Eigen::Translation3f{delta.normalized() * offset});
  }
  return source;
}

}  // namespace gen
}  // namespace cho

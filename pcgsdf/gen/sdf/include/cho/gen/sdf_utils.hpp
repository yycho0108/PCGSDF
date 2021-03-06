#pragma once

#include "cho/gen/sdf.hpp"

namespace cho {
namespace gen {

float ComputeSdfVolumeMonteCarlo(const cho::gen::SdfPtr &sdf,
                                 const int num_samples);

cho::gen::SdfPtr MakeTangentApprox(const cho::gen::SdfPtr &source,
                                   const cho::gen::SdfPtr &target,
                                   const bool force_tangent = false);

template <typename Iterator>
cho::gen::SdfPtr UnionTree(Iterator i0, Iterator i1) {
  const int d = std::distance(i0, i1);
  if (d <= 1) {
    return *i0;
  }
  auto im = std::next(i0, d / 2);
  return cho::gen::OpUnion::Create(UnionTree(i0, im), UnionTree(im, i1));
}

template <typename Lhs, typename Rhs>
cho::gen::SdfPtr Union(const Lhs &lhs, const Rhs &rhs) {
  if (!lhs) {
    return rhs;
  }
  if (!rhs) {
    return lhs;
  }
  return OpUnion::Create(lhs, rhs);
}

}  // namespace gen
}  // namespace cho

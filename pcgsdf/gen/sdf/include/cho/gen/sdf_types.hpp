
#pragma once

#include <vector>

#include "cho/gen/sdf_fwd.hpp"

namespace cho {
namespace gen {
enum class SdfOpCode : std::int8_t {
  SPHERE,
  BOX,
  CYLINDER,
  PLANE,
  TORUS,
  CONE,
  HEX_PRISM,
  OCTAHEDRON,
  PYRAMID,

  ROUND,
  NEGATION,
  UNION,
  INTERSECTION,
  SUBTRACTION,
  ONION,
  TRANSLATION,
  ROTATION,
  TRANSFORMATION,
  SCALE_BEGIN,
  SCALE_END,
};

struct SdfData {
  SdfOpCode code;
  std::vector<float> param;
};

}  // namespace gen
}  // namespace cho

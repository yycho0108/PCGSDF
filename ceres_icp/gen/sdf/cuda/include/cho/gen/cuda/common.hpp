#pragma once

#include <cstdlib>

#include "cho/gen/sdf_fwd.hpp"

struct SdfDataCompact {
  cho::gen::SdfOpCode code;
  float* param;
};

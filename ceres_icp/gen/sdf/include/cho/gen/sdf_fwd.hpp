#pragma once

#include <memory>

namespace cho {
namespace gen {

struct SdfData;

class SdfInterface;
using SdfPtr = std::shared_ptr<SdfInterface>;
using SdfConstPtr = std::shared_ptr<const SdfInterface>;

}  // namespace gen
}  // namespace cho

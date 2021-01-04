#pragma once

#include <memory>

namespace cho {
namespace gen {


class SdfInterface;
using SdfPtr = std::shared_ptr<SdfInterface>;
using SdfConstPtr = std::shared_ptr<const SdfInterface>;

enum class SdfOpCode : std::int8_t;

struct SdfData;

}  // namespace gen
}  // namespace cho

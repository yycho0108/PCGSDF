#include "cho/gen/gen_utils.hpp"

#include <algorithm>
#include <random>
#include <variant>

#include <Eigen/Geometry>

#include "cho/gen/sdf.hpp"
#include "cho/gen/sdf_utils.hpp"

namespace cho {
namespace gen {
cho::gen::SdfPtr GenerateSpace(std::default_random_engine &rng,
                               const SpaceOptions &opts,
                               const Eigen::Vector3f &eye,
                               cho::gen::SdfPtr *const vol) {
  using namespace cho::gen;

  // Generate number of boxes.
  std::uniform_int_distribution ndist{opts.min_num_boxes, opts.max_num_boxes};
  const int num_boxes = ndist(rng);

  // TODO(ycho): Expose these parameters.
  std::uniform_real_distribution<float> dist{opts.min_box_size,
                                             opts.max_box_size};

  std::vector<SdfPtr> rooms;
  rooms.reserve(num_boxes);
  SdfPtr out;
  while (rooms.size() < num_boxes) {
    // Generate a "room" (a box, really)
    std::array<float, traits<Box>::DoF> a;
    std::generate(a.begin(), a.end(), [&dist, &rng]() { return dist(rng); });
    auto room = Box::CreateFromArray(a);

    // Conditionally adjust size.
    if (room->Radius() < dist.min() || room->Radius() > dist.max()) {
      const float target_radius = dist(rng);
      room = Scale::Create(room, target_radius / room->Radius());
    }

    // Apply a random transform.
    const Eigen::Vector3f pos{dist(rng), dist(rng), dist(rng)};
    room = OpTransformation::Create(room, Eigen::Translation3f{pos});

    if (rooms.empty()) {
      // Initial room must contain camera.
      // TODO(ycho): Relax this arbitrary constraint.
      if (room->Distance(eye) > 0) {
        continue;
      }
      out = room;
      rooms.emplace_back(room);
    } else {
      room = MakeTangentApprox(room, out, false);
      out = OpUnion::Create(out, room);
      rooms.emplace_back(room);
    }
  }

  // Rebuild `out` from balanced tree of unions.
  out = UnionTree(rooms.begin(), rooms.end());

  // Optionally export internal volume.
  if (vol) {
    *vol = out;
  }

  // Convert to a "wall" to create a valid geometry.
  // This gives thickness on our model of the negative space.
  out = OpOnion::Create(out, 0.001f);
  return out;
}

namespace detail {

template <int N, typename... Ts>
using NthTypeOf = typename std::tuple_element<N, std::tuple<Ts...>>::type;

template <typename... Types>
constexpr int Count() {
  return sizeof...(Types);
}

template <typename Sdf, typename Dist, typename Rng>
cho::gen::SdfPtr GeneratePrimitive(Dist &dist, Rng &rng) {
  std::array<float, traits<Sdf>::DoF> a;
  std::generate(a.begin(), a.end(), [&dist, &rng]() { return dist(rng); });
  return Sdf::CreateFromArray(a);
}

template <typename... Types, typename Dist, typename Rng, std::size_t... Is>
cho::gen::SdfPtr GeneratePrimitiveFromIndexImpl(const std::size_t which,
                                                Dist &dist, Rng &rng,
                                                std::index_sequence<Is...>) {
  cho::gen::SdfPtr out;
  std::initializer_list<int>(
      {(which == Is
            ? ((void)(out = detail::GeneratePrimitive<NthTypeOf<Is, Types...>>(
                          dist, rng)),
               0)
            : 0)...});
  if (!out) {
    throw std::out_of_range(fmt::format("invalid index = {}\n", which));
  }
  return out;
}

}  // namespace detail

template <typename... Ts, typename Dist, typename Rng>
cho::gen::SdfPtr GeneratePrimitiveFromIndex(const int which, Dist &dist,
                                            Rng &rng) {
  return detail::GeneratePrimitiveFromIndexImpl<Ts...>(
      which, dist, rng, std::make_index_sequence<sizeof...(Ts)>{});
}

template <typename... Ts>
class SdfPrimitiveDistribution {
 public:
  explicit SdfPrimitiveDistribution(const float vmin, const float vmax)
      : value_dist_{vmin, vmax}, type_dist_{0, detail::Count<Ts...>() - 1} {}

  template <typename Rng>
  SdfPtr operator()(Rng &rng) {
    const int which = type_dist_(rng);
    auto out = GeneratePrimitiveFromIndex<Ts...>(which, value_dist_, rng);
    if (!out) {
      fmt::print("ERROR : Sdf was not produced @ which = {}\n", which);
    }
    return out;
  }

 private:
  std::uniform_real_distribution<float> value_dist_;
  std::uniform_int_distribution<int> type_dist_;
};

cho::gen::SdfPtr GenerateObject(std::default_random_engine &rng,
                                const ObjectOptions &opts,
                                const cho::gen::SdfPtr &scene,
                                const cho::gen::SdfPtr &scene_vol,
                                const Eigen::Vector3f &eye) {
  using namespace cho::gen;

  // Generate number of primitives.
  std::uniform_int_distribution ndist{opts.min_num_primitives,
                                      opts.max_num_primitives};
  const int num_primitives = ndist(rng);

  SdfPtr out{nullptr};
  const float sr = scene->Radius();
  const Eigen::AlignedBox3f scene_box{
      scene->Center() - sr * Eigen::Vector3f::Ones(),
      scene->Center() + sr * Eigen::Vector3f::Ones()};

  // NOTE(ycho): Generally, value_dist ~ geometry dimensions.
  SdfPrimitiveDistribution<Sphere, Box, Cylinder, Cone, Torus> sdf_dist{-2.0F,
                                                                        2.0F};

  std::vector<SdfPtr> parts;
  while (parts.size() < num_primitives) {
    // Generate primitive.
    // TODO(yycho0108): Decay primitive size as a function of hierarchy?
    SdfPtr sdf = sdf_dist(rng);

    // Initialize ...
    if (!out) {
      // Sample initial position from ~approx free space in scene
      Eigen::Vector3f pos = scene_box.sample();

      // If volume information is available, sample from "inside"
      if (scene_vol) {
        do {
          pos = scene_box.sample();
        } while (scene_vol->Distance(pos) > 0);
      }

      sdf = OpTransformation::Create(
          sdf, Eigen::Translation3f{pos} * Eigen::Quaternionf::UnitRandom());

      // Reject sdfs that include `eye` ...
      if (sdf->Distance(eye) <= 0) {
        continue;
      }
      out = sdf;
      parts.emplace_back(sdf);
      continue;
    }

    // Generate "reasonable" translation.
    const float rR = out->Radius() + sdf->Radius();
    const Eigen::AlignedBox3f box{Eigen::Vector3f{-rR, -rR, -rR},
                                  Eigen::Vector3f{+rR, +rR, +rR}};
    Eigen::Vector3f offset;
    do {
      offset = box.sample();
    } while (offset.squaredNorm() >= rR * rR);
    const Eigen::Vector3f pos = out->Center() + offset;

    // Apply transform.
    sdf = OpTransformation::Create(
        sdf, Eigen::Translation3f{pos} * Eigen::Quaternionf::UnitRandom());
    if (sdf->Distance(eye) <= 0) {
      continue;
    }

    // debug-only construct: connect node hierarchy.
    // if (!dbg) {
    //  auto d = out->Center() - sdf->Center();
    //  auto cyl = Cylinder::Create(0.5 * d.norm(), 0.1f);
    //  cyl = Transformation::Create(
    //      cyl, Eigen::Translation3f{0.5 * (out->Center() + sdf->Center())} *
    //               Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(),
    //                                                  d.normalized()));
    //  dbg = cyl;
    //} else {
    //  auto d = out->Center() - sdf->Center();
    //  auto cyl = Cylinder::Create(0.5 * d.norm(), 0.1f);
    //  cyl = Transformation::Create(
    //      cyl, Eigen::Translation3f{0.5 * (out->Center() + sdf->Center())} *
    //               Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(),
    //                                                  d.normalized()));
    //  dbg = Union::Create(dbg, cyl);
    //}

    // Merge.
    sdf = MakeTangentApprox(sdf, out);
    out = OpUnion::Create(out, sdf);
    parts.emplace_back(sdf);
    // out = SmoothUnion::Create(out, sdf, 0.25f);
  }

  out = UnionTree(parts.begin(), parts.end());

  // if (dbg) {
  //  out = Union::Create(out, dbg);
  //}
  return out;
}

cho::gen::SdfPtr GenerateScene(std::default_random_engine &rng,
                               const Eigen::Vector3f &eye,
                               const SceneOptions &opts,
                               cho::gen::SdfPtr *const objs_out,
                               cho::gen::SdfPtr *const free_space,
                               cho::gen::SdfPtr *const wall) {
  using namespace cho::gen;

  // Generate Space.
  SdfPtr vol{nullptr};
  auto space = cho::gen::GenerateSpace(rng, opts.space_opts, eye, &vol);

  // FIXME(ycho): Remove these hacks.
  // Export some partial sdfs...
  if (free_space) {
    *free_space = vol;
  }
  if (wall) {
    *wall = space;
  }

  // Generate Objects.
  std::uniform_int_distribution ndist{opts.max_num_objects,
                                      opts.max_num_objects};
  const int num_objects = ndist(rng);
  std::vector<SdfPtr> objects;
  SdfPtr objs;
  for (int i = 0; i < num_objects; ++i) {
    auto obj = cho::gen::GenerateObject(rng, opts.object_opts, space, vol, eye);
    objects.emplace_back(obj);
  }
  objs = UnionTree(objects.begin(), objects.end());

  // Optionally output objects separately.
  if (objs_out) {
    *objs_out = objs;
  }

  SdfPtr scene = OpUnion::Create(space, objs);
  return scene;
}

}  // namespace gen
}  // namespace cho

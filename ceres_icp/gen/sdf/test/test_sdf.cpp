#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include <fmt/printf.h>
#include <Eigen/Geometry>

#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/visualization/utility/DrawGeometry.h>
#include <open3d/visualization/visualizer/Visualizer.h>

#include "cho/gen/ray_marching.hpp"
#include "cho/gen/sdf.hpp"

#include "cho/gen/cuda/render.hpp"
// #include "cho/gen/cuda/render_jit.hpp"

std::default_random_engine rng;

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
                                   const bool force_tangent = false) {
  const Eigen::Vector3f delta = target->Center() - source->Center();
  const float d1 = source->Distance(target->Center());
  const float d2 = target->Distance(source->Center());
  const float offset = (d1 + d2 - delta.norm());
  // Don't `maketangent` if already intersecting
  if (force_tangent || offset > 0) {
    return cho::gen::Transformation::Create(
        source, Eigen::Translation3f{delta.normalized() * offset});
  }
  return source;
}

template <typename Iterator>
cho::gen::SdfPtr UnionTree(Iterator i0, Iterator i1) {
  const int d = std::distance(i0, i1);
  if (d <= 1) {
    return *i0;
  }
  auto im = std::next(i0, d / 2);
  return cho::gen::Union::Create(UnionTree(i0, im), UnionTree(im, i1));
}

cho::gen::SdfPtr GenerateSpace(const int num_boxes, const Eigen::Vector3f &eye,
                               cho::gen::SdfPtr *const vol) {
  using namespace cho::gen;
  std::uniform_real_distribution<float> sdist{5.0, 15.0};
  std::uniform_real_distribution<float> udist{-10.0, 10.0};

  std::vector<SdfPtr> rooms;
  rooms.reserve(num_boxes);
  SdfPtr out;
  while (rooms.size() < num_boxes) {
    std::array<float, traits<Box>::DoF> a;
    std::generate(a.begin(), a.end(), [&udist]() { return udist(rng); });
    auto room = Box::CreateFromArray(a);
    // fmt::print("Room radius = {}\n", room->Radius());

    // Adjust size.
    const float target_radius = sdist(rng);
    room = Scale::Create(room, target_radius / room->Radius());

    // Transform ...
    const Eigen::Vector3f pos{udist(rng), udist(rng), udist(rng)};
    room = Transformation::Create(room, Eigen::Translation3f{pos});

    if (rooms.empty()) {
      // Initial room must contain camera.
      if (room->Distance(eye) > 0) {
        continue;
      }
      out = room;
      rooms.emplace_back(room);
    } else {
      room = MakeTangentApprox(room, out, false);
      out = Union::Create(out, room);
      rooms.emplace_back(room);
    }
  }

  // Rebuild out from balanced tree of unions.
  out = UnionTree(rooms.begin(), rooms.end());

  // out = Sphere::Create(30.0F);
  // out = Plane::Create(Eigen::Vector3f::UnitZ(), 1.0F);
  // out = Box::Create(Eigen::Vector3f{5.0, 5.0, 5.0});

  // vol == occupied internal volume
  if (vol) {
    *vol = out;
  }

  // Convert to a "wall" to create a valid geometry.
  // This gives thickness on our model of the negative space.
  out = Onion::Create(out, 0.0001f);

  return out;
}

cho::gen::SdfPtr GenerateObject(const int num_primitives,
                                const cho::gen::SdfPtr &scene,
                                const cho::gen::SdfPtr &scene_vol,
                                const Eigen::Vector3f &eye) {
  using namespace cho::gen;

  SdfPtr out{nullptr};

  const float sr = scene->Radius();
  // const float sr = 10.0F;
  const Eigen::AlignedBox3f scene_box{
      scene->Center() - sr * Eigen::Vector3f::Ones(),
      scene->Center() + sr * Eigen::Vector3f::Ones()};

  // NOTE(yycho0108): Object radius ~ 0.1x scene radius
  // std::uniform_real_distribution<float> udist{-0.1F * sr, 0.1F * sr};
  std::uniform_real_distribution<float> udist{-2, 2};
  std::uniform_int_distribution<int> sdf_type{0, 2};

  std::vector<SdfPtr> parts;
  while (parts.size() < num_primitives) {
    // Generate primitive.
    // TODO(yycho0108): Decay primitive size as a function of hierarchy?
    auto t = sdf_type(rng);
    // t = 0;
    SdfPtr sdf;
    switch (t) {
      case 0: {
        std::array<float, traits<Sphere>::DoF> a;
        std::generate(a.begin(), a.end(), [&udist]() { return udist(rng); });
        sdf = Sphere::CreateFromArray(a);
        break;
      }
      case 1: {
        std::array<float, traits<Box>::DoF> a;
        std::generate(a.begin(), a.end(), [&udist]() { return udist(rng); });
        sdf = Box::CreateFromArray(a);
        break;
      }
      case 2: {
        std::array<float, traits<Cylinder>::DoF> a;
        std::generate(a.begin(), a.end(), [&udist]() { return udist(rng); });
        sdf = Cylinder::CreateFromArray(a);
        break;
      }
      default: {
        throw std::out_of_range("out of range!");
        break;
      }
    }

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

      sdf = Transformation::Create(
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
    sdf = Transformation::Create(
        sdf, Eigen::Translation3f{pos} * Eigen::Quaternionf::UnitRandom());
    if (sdf->Distance(eye) <= 0) {
      continue;
    }

    // debug-only construct .
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
    out = Union::Create(out, sdf);
    parts.emplace_back(sdf);
    // out = SmoothUnion::Create(out, sdf, 0.25f);
  }

  out = UnionTree(parts.begin(), parts.end());

  // if (dbg) {
  //  out = Union::Create(out, dbg);
  //}
  return out;
}

cho::gen::SdfPtr CreateDefaultScene(const Eigen::Vector3f &eye,
                                    const int num_objects,
                                    cho::gen::SdfPtr *const objs_out) {
  using namespace cho::gen;

#if 0
  auto room = Box::Create(Eigen::Vector3f{10, 10, 10});
  SdfPtr vol = room;
  // room = Transformation::Create(room, Eigen::Translation3f{ 0, 0, 10 });
  room = Onion::Create(room, 0.1f);
  // room = Negation::Create(room, 0.1f);
  fmt::print("RC = {}\n", room->Center().transpose());
#else
  SdfPtr vol{nullptr};
  auto room = GenerateSpace(5, eye, &vol);
#endif

  // auto ground = Plane::Create(Eigen::Vector3f::UnitZ(), 0.0F);
  // auto wall0 = Plane::Create(-Eigen::Vector3f::UnitX(), 5.0F);
  // auto wall1 = Plane::Create(Eigen::Vector3f::UnitX(), 5.0F);
  // auto wall2 = Plane::Create(-Eigen::Vector3f::UnitY(), 2.0F);
  // auto wall3 = Plane::Create(Eigen::Vector3f::UnitY(), 2.0F);

  auto pillar0 = Cylinder::Create(2.5, 0.15f);
  pillar0 = Transformation::Create(pillar0, Eigen::Translation3f{2.0, 0, 2.5});

  // return GenerateObject(6, eye);

  SdfPtr scene{room};
  // SdfPtr scene{ ground };
  // scene = Union::Create(scene, wall0);
  // scene = Union::Create(scene, wall1);
  // scene = Union::Create(scene, wall2);
  // scene = Union::Create(scene, wall3);
  // scene = Union::Create(scene, pillar0);
  // return scene;

  // Generate Objects.
  std::uniform_int_distribution ndist{1, 8};
  std::vector<SdfPtr> objects;
  SdfPtr objs;
  for (int i = 0; i < num_objects; ++i) {
    auto obj = GenerateObject(ndist(rng), scene, vol, eye);
    objects.emplace_back(obj);
  }
  objs = UnionTree(objects.begin(), objects.end());
  // [Optional] output objects only
  if (objs_out) {
    *objs_out = objs;
  }
  scene = Union::Create(scene, objs);
  // scene = objs;
  return scene;
}

Eigen::MatrixXf CreateDepthImage(const Eigen::Isometry3f &camera_pose,
                                 const Eigen::Vector2i &resolution,
                                 const Eigen::Vector2f &fov,
                                 const cho::gen::SdfPtr &scene,
                                 std::vector<Eigen::Vector3f> *const cloud,
                                 const bool compile = false) {
  Eigen::MatrixXf out(resolution(0), resolution(1));

  auto sdf = cho::gen::Transformation::Create(scene, camera_pose.inverse());
  std::vector<cho::gen::SdfData> prog;
  if (compile) {
    sdf->Compile(&prog);
  }

  const Eigen::Vector2f step =
      fov.array() / (resolution.array() - 1).cast<float>();

  float ang_v{-fov.x() / 2};
  if (cloud) {
    cloud->clear();
    cloud->reserve(resolution.prod());
  }
  for (int i = 0; i < resolution.x(); ++i) {
    float ang_h = -fov.y() / 2;
    for (int j = 0; j < resolution.y(); ++j) {
      const Eigen::Vector3f ray =
          Eigen::AngleAxisf{ang_h, Eigen::Vector3f::UnitZ()} *
          Eigen::AngleAxisf{ang_v, Eigen::Vector3f::UnitY()} *
          Eigen::Vector3f::UnitX();
      float depth;
      if (compile) {
        depth = cho::gen::RayMarchingDepthWithProgram(Eigen::Vector3f::Zero(),
                                                      ray, prog, 128);
      } else {
        depth =
            cho::gen::RayMarchingDepth(Eigen::Vector3f::Zero(), ray, sdf, 128);
      }

      out.coeffRef(i, j) = depth;
      if (cloud) {
        cloud->emplace_back(depth * ray);
      }
      ang_h += step.y();
    }
    ang_v += step.x();
  }
  return out;
}

auto O3dCloudFromVecVector3f(const std::vector<Eigen::Vector3f> &cloud_in) {
  std::vector<Eigen::Vector3f> cloud_f = cloud_in;
  if (true) {
    cloud_f.erase(std::remove_if(cloud_f.begin(), cloud_f.end(),
                                 [](const Eigen::Vector3f &v) -> bool {
                                   return !v.allFinite();
                                   // || (v.array().abs() >= 100).any();
                                 }),
                  cloud_f.end());
  }
  std::vector<Eigen::Vector3d> cloud_d(cloud_f.size());
  std::transform(cloud_f.begin(), cloud_f.end(), cloud_d.begin(),
                 [](const Eigen::Vector3f &v) -> Eigen::Vector3d {
                   return v.cast<double>();
                 });
  // open3d::geometry::PointCloud cloud_o3d;
  auto cloud_o3d = std::make_shared<open3d::geometry::PointCloud>(cloud_d);
  return cloud_o3d;
}

int main() {
  // Configure ...
  static constexpr const float kDegree{M_PI / 180.0F};
  static constexpr const int kVerticalResolution{128};
  static constexpr const int kHorizontalResolution{128};
  static constexpr const float kVerticalFov{120 * kDegree};
  static constexpr const float kHorizontalFov{210 * kDegree};
  static constexpr const int kNumObjects{8};
  static constexpr const bool kShow{true};

  // Initialize RNG.
  // const std::int64_t seed =
  //    std::chrono::high_resolution_clock::now().time_since_epoch().count();
  // const std::int64_t seed = 0;
  const std::int64_t seed = 1609268978845759618;
  fmt::print("seed={}\n", seed);
  rng.seed(seed);

  // Create camera.
  Eigen::Vector2i resolution{kVerticalResolution, kHorizontalResolution};
  Eigen::Vector2f fov{kVerticalFov, kHorizontalFov};
  Eigen::Vector3f eye{0, 0, 0};
  Eigen::AngleAxisf eye_rot{0 * kDegree, Eigen::Vector3f::UnitY()};
  Eigen::Isometry3f eye_pose{Eigen::Translation3f{eye} * eye_rot};

  // Create scene - take in camera parameter to avoid collision.
  fmt::print("Scene Generation Start.\n");
  cho::gen::SdfPtr objs;
  auto scene_sdf = CreateDefaultScene(eye, kNumObjects, &objs);
  fmt::print("Scene Generation End.\n");

  // Raycast...
  std::vector<Eigen::Vector3f> cloud_f;
  std::vector<Eigen::Vector3f> cloud_f_c;
  std::vector<Eigen::Vector3f> cloud_f_cu;

  Eigen::MatrixXf depth_image, depth_image_c, depth_image_cu;

  auto t0 = std::chrono::high_resolution_clock::now();
  if (false) {
    depth_image =
        CreateDepthImage(eye_pose, resolution, fov, scene_sdf, &cloud_f);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  if (false) {
    depth_image_c = CreateDepthImage(eye_pose, resolution, fov, scene_sdf,
                                     &cloud_f_c, true);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  // CUDA
  std::vector<cho::gen::SdfData> program;
  scene_sdf->Compile(&program);
  Eigen::AngleAxisf q{0.1, Eigen::Vector3f::UnitZ()};
  Eigen::Isometry3f eye_pose_q{eye_pose};
#if 1
  for (int k = 0; k < 1; ++k) {
    eye_pose_q = eye_pose_q * q;
    CreateDepthImageCuda(eye_pose_q, resolution, fov, program, &depth_image_cu,
                         &cloud_f_cu);
  }
#endif
  auto t3 = std::chrono::high_resolution_clock::now();

#if 1
  {
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow();

    auto cloud = O3dCloudFromVecVector3f(cloud_f_cu);
    vis.AddGeometry(cloud);

    // axes
    auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(
        0.5, eye.cast<double>());
    vis.AddGeometry(axes);

    Eigen::Isometry3f eye_pose_q{eye_pose};
    Eigen::AngleAxisf q{0.1, Eigen::Vector3f::UnitZ()};

    // for (int k = 0; k < 128; ++k) {
    vis.GetViewControl().SetConstantZFar(100.0F);
    vis.GetViewControl().SetConstantZNear(0.001F);
    // vis.GetViewControl().SetZoom(1.0);

    open3d::camera::PinholeCameraParameters p;
    vis.GetViewControl().ConvertToPinholeCameraParameters(p);
    p.extrinsic_.topRightCorner<3, 1>() = eye.cast<double>();
    vis.GetViewControl().ConvertFromPinholeCameraParameters(p);

    Eigen::Quaternionf q_axes = Eigen::Quaternionf::Identity();
    float tsum = 0.0;
    int tcount{0};
    while (true) {
      // Estimate FPS.
      auto t0 = std::chrono::high_resolution_clock::now();
      if (tcount > 0) {
        const float tmean = tsum / tcount / 1e6;
        const float fps = 1.0 / tmean;
        fmt::print("fps = {}\n", fps);
      }

      open3d::camera::PinholeCameraParameters p;
      vis.GetViewControl().ConvertToPinholeCameraParameters(p);

      const Eigen::Isometry3f optical_from_world{p.extrinsic_.cast<float>()};

      Eigen::Isometry3f optical_from_camera = Eigen::Isometry3f::Identity();
      optical_from_camera =
          Eigen::AngleAxisf(-90 * kDegree, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(90 * kDegree, Eigen::Vector3f::UnitX()) *
          Eigen::Translation3f{0.0, 0.0, 0.0};

      const Eigen::Isometry3f world_from_camera =
          (optical_from_world.inverse() * optical_from_camera);
      eye_pose_q = world_from_camera;

      // 1.4x factor to account for distortion

      // p.intrinsic_.
      fov.array() = static_cast<float>(
          1.0 * vis.GetViewControl().GetFieldOfView() * kDegree);
      // fov.array() = 90.0F * kDegree;

#if 1
      CreateDepthImageCuda(eye_pose_q, resolution, fov, program,
                           &depth_image_cu, &cloud_f_cu);
#else
      depth_image_cu =
          CreateDepthImage(eye_pose_q, resolution, fov, scene_sdf, &cloud_f_cu);
#endif

      // hmm?
      auto cloud2 = O3dCloudFromVecVector3f(cloud_f_cu);
      cloud->points_ = cloud2->points_;

#if 0
      // Where the hell is the camera??
      axes->Translate(world_from_camera.translation().cast<double>(), false);

      const Eigen::Quaternionf delta =
          Eigen::Quaternionf{world_from_camera.linear().cast<float>()} *
          q_axes.inverse();
      q_axes = delta * q_axes;
      axes->Rotate(delta.toRotationMatrix().cast<double>(),
                   world_from_camera.translation().cast<double>());
      vis.UpdateGeometry(axes);
#endif

      vis.UpdateGeometry(cloud);
      vis.PollEvents();
      vis.UpdateRender();

      auto t1 = std::chrono::high_resolution_clock::now();

      tsum += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
      ++tcount;
    }
    return 0;
  }
#endif

#if 0
  // CUDA+JIT
  Eigen::MatrixXf depth_image_cu_jit;
  std::vector<Eigen::Vector3f> cloud_f_cu_jit;
  const std::string point_arg{"point"};
  std::string prefix{"jit_tmp"};
  std::string subex{""};
  int count{0};
  const std::string program_source =
      scene_sdf->Jit(point_arg, prefix, &count, &subex);
  fmt::print("{}\n const float distance = {};\n", subex, program_source);

  Eigen::AngleAxisf q{0.1, Eigen::Vector3f::UnitZ()};
  for (int k = 0; k < 128; ++k) {
    const Eigen::Isometry3f eye_pose_q = eye_pose * q;
    CreateDepthImageCudaJit(eye_pose_q, resolution, fov, point_arg, subex,
                            program_source, &depth_image_cu_jit,
                            &cloud_f_cu_jit);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
#endif
  fmt::print(
      "R={} v C={} v CU={}\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count(),
      std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count());

  cloud_f = cloud_f_cu;

  std::vector<Eigen::Vector3f> cloud_f_obj;
  Eigen::MatrixXf depth_image_obj;
  if (0) {
    depth_image_obj =
        CreateDepthImage(eye_pose, resolution, fov, objs, &cloud_f_obj);
  }

  // Direct visualization ...
  if (kShow) {
    if (true) {
      cloud_f.erase(std::remove_if(cloud_f.begin(), cloud_f.end(),
                                   [](const Eigen::Vector3f &v) -> bool {
                                     return !v.allFinite() ||
                                            (v.array().abs() >= 100).any();
                                   }),
                    cloud_f.end());
    }
    std::vector<Eigen::Vector3d> cloud_d(cloud_f.size());
    std::transform(cloud_f.begin(), cloud_f.end(), cloud_d.begin(),
                   [](const Eigen::Vector3f &v) -> Eigen::Vector3d {
                     return v.cast<double>();
                   });
    // open3d::geometry::PointCloud cloud_o3d;
    auto cloud_o3d = std::make_shared<open3d::geometry::PointCloud>(cloud_d);
    auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(
        0.5, eye.cast<double>());

    open3d::visualization::DrawGeometries({cloud_o3d, axes});
  }

  if (true) {
    auto export_data = [](const std::vector<Eigen::Vector3f> &cloud_,
                          const Eigen::MatrixXf &depth_image,
                          const std::string &prefix) {
      // Cleanup cloud.
      std::vector<Eigen::Vector3f> cloud = cloud_;
      cloud.erase(std::remove_if(cloud.begin(), cloud.end(),
                                 [](const Eigen::Vector3f &v) -> bool {
                                   return !v.allFinite();
                                   // return !v.allFinite() ||
                                   //       (v.array().abs() >= 10).any();
                                 }),
                  cloud.end());

      // Export cloud.
      std::ofstream fout_cloud(fmt::format("/tmp/{}-cloud.csv", prefix));
      for (const auto &v : cloud) {
        fout_cloud << v.transpose() << std::endl;
      }

      // Export depth.
      std::ofstream fout_depth(fmt::format("/tmp/{}-depth.csv", prefix));
      for (int i = 0; i < depth_image.rows(); ++i) {
        for (int j = 0; j < depth_image.cols(); ++j) {
          fout_depth << depth_image.coeff(i, j) << ' ';
        }
        fout_depth << std::endl;
      }
    };

    export_data(cloud_f, depth_image, "scene");
    export_data(cloud_f_c, depth_image_c, "scene-compiled");
    export_data(cloud_f_obj, depth_image_obj, "object");
  }

  return 0;
}

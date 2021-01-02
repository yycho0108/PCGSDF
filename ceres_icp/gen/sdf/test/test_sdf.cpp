#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include <fmt/printf.h>
#include <Eigen/Geometry>

#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/visualization/utility/DrawGeometry.h>
// #include <open3d/visualization/visualizer/Visualizer.h>
#include <open3d/visualization/visualizer/VisualizerWithKeyCallback.h>

// Only for key codes
#include <GLFW/glfw3.h>

#include "cho/gen/cuda/render.hpp"
#include "cho/gen/gen_utils.hpp"
#include "cho/gen/ray_marching.hpp"
#include "cho/gen/sdf.hpp"
#include "cho/gen/sdf_utils.hpp"
// #include "cho/gen/cuda/render_jit.hpp"

std::default_random_engine rng;

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

struct TrajectoryGenerationOptions {
  int max_search_steps;
  int max_depth;
  float max_length;

  float max_edge_length;
  float max_edge_angle;

  float max_search_time{5000.0F};
};

namespace detail {
bool GenerateTrajectoryImpl(const cho::gen::SdfPtr &space,
                            const TrajectoryGenerationOptions &opts,
                            const Eigen::AlignedBox3f &aabb,
                            const float cur_length,
                            std::vector<Eigen::Isometry3f> *const trajectory) {
  auto t_start = std::chrono::high_resolution_clock::now();
  static constexpr const float kMaxDistance{0.5F};

  // Check for success.
  const bool is_long = cur_length >= opts.max_length;
  if (is_long || trajectory->size() >= opts.max_depth) {
    return is_long;
  }

  // Prune "impossible" cases.
  const int num_remaining_edges = (opts.max_depth - trajectory->size());
  const float max_future_length =
      cur_length + opts.max_edge_length * num_remaining_edges;
  if (max_future_length < opts.max_length) {
    return false;
  }

  // Create options for the child....
  TrajectoryGenerationOptions child_opts = opts;

  std::uniform_real_distribution<float> udist{0.0, 1.0};
  for (int i = 0; i < opts.max_search_steps; ++i) {
    // Apply timeout and abort.
    auto t_now = std::chrono::high_resolution_clock::now();
    const int elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_start)
            .count();
    const float time_left = opts.max_search_time - elapsed;
    if (time_left < 0) {
      return false;
    }
    const int child_left = (opts.max_search_steps - i);
    child_opts.max_search_time = time_left / child_left;

    // Generate Waypoint.
    Eigen::Vector3f p;
    Eigen::Quaternionf q;
    bool wpt_found{false};
    for (int j = 0; j < opts.max_search_steps; ++j) {
      // Generate orientation.
      if (trajectory->empty()) {
        q = Eigen::Quaternionf::UnitRandom();
      } else {
        const Eigen::Vector3f rpy =
            opts.max_edge_angle * (Eigen::Array3f::Random() - 0.5F);
        q = trajectory->back().linear() *
            Eigen::AngleAxisf{rpy.x(), Eigen::Vector3f::UnitX()} *
            Eigen::AngleAxisf{rpy.y(), Eigen::Vector3f::UnitY()} *
            // NOTE(yycho0108): Reduce roll component since it's so
            // disorienting...
            // TODO(ycho): Fix hardcoded adjustment.
            Eigen::AngleAxisf{0.25F * rpy.z(), Eigen::Vector3f::UnitZ()};
      }

      auto delta = Eigen::AlignedBox3f{
          opts.max_edge_length * Eigen::Array3f{0, -0.25, -0.25},
          opts.max_edge_length * Eigen::Array3f{1, 0.25, 0.25}};
      const Eigen::Vector3f v = delta.sample();
      const Eigen::Vector3f &nxt_rel =
          v.norm() >= opts.max_edge_length
              ? v * udist(rng) * opts.max_edge_length / v.norm()
              : v;

      p = trajectory->empty() ? aabb.sample() : trajectory->back() * nxt_rel;

      // NOTE(yycho0108): `dmin` here is a somewhat arbitrary requirement
      // that allows trajectory in the linear interpolation from the previous to
      // current point will not be occupied.
      // TODO(yycho0108): instead of hardcoding `kMaxDistance`,
      // Precompute a heuristically determined "spacious" distance from N random
      // samples.
      const float dmin = trajectory->empty()
                             ? kMaxDistance * (opts.max_search_steps - j) /
                                   opts.max_search_steps
                             : (trajectory->back().translation() - p).norm();
      // Alternative formulation:
      // Prefer a somewhat spacious locale, but decay this preference
      // over search iteration. (currently linearly decayed)
      // const float target_distance = std::max(
      //    dmin, kMaxDistance * (opts.max_search - j) / opts.max_search);
      const float target_distance = std::max(0.0F, 1.0F * dmin);
      if (space->Distance(p) >= -target_distance) {
        continue;
      }

      // Found waypoint : exit loop.
      wpt_found = true;
      break;
    }

    if (!wpt_found) {
      continue;
    }

    // Try to go down this route.
    const float edge_length =
        trajectory->empty() ? 0.0F
                            : (p - trajectory->back().translation()).norm();
    trajectory->emplace_back(Eigen::Translation3f{p} * q);

    // Return if success.
    if (GenerateTrajectoryImpl(space, child_opts, aabb,
                               cur_length + edge_length, trajectory)) {
      return true;
    }
    trajectory->pop_back();
  }
  return false;
}
}  // namespace detail

bool GenerateTrajectory(const cho::gen::SdfPtr &space,
                        const TrajectoryGenerationOptions &opts,
                        std::vector<Eigen::Isometry3f> *const trajectory) {
  const Eigen::AlignedBox3f aabb{space->Center().array() - space->Radius(),
                                 space->Center().array() + space->Radius()};
  return detail::GenerateTrajectoryImpl(space, opts, aabb, 0.0F, trajectory);
}

cho::gen::SdfPtr TrajectoryToSdf(
    const std::vector<Eigen::Isometry3f> &trajectory) {
  if (trajectory.size() < 2) {
    return nullptr;
  }

  std::vector<cho::gen::SdfPtr> sdfs;
  sdfs.reserve(trajectory.size() - 1);

  auto it_prv = trajectory.begin();
  for (auto it = std::next(it_prv); it != trajectory.end(); ++it) {
    const auto &prv = *it_prv;
    const auto &cur = *it;

    const Eigen::Vector3f delta = cur.translation() - prv.translation();

    const Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(
        Eigen::Vector3f::UnitZ(), delta.normalized());
    fmt::print("Q = {} {} {} {}\n", q.x(), q.y(), q.z(), q.w());
    const Eigen::Vector3f t = 0.5F * (cur.translation() + prv.translation());

    auto sdf = cho::gen::Cylinder::Create(0.5F * delta.norm(), 0.1F);
    sdf = cho::gen::Transformation::Create(sdf, Eigen::Translation3f{t} * q);
    sdfs.emplace_back(sdf);
    it_prv = it;
  }
  return UnionTree(sdfs.begin(), sdfs.end());
}

Eigen::Isometry3f GetCameraPose(open3d::visualization::Visualizer *const v) {
  static const Eigen::Quaternionf optical_from_camera =
      Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f::UnitX());

  open3d::camera::PinholeCameraParameters p;
  v->GetViewControl().ConvertToPinholeCameraParameters(p);

  const Eigen::Isometry3f optical_from_world{p.extrinsic_.cast<float>()};
  const Eigen::Isometry3f world_from_camera =
      (optical_from_world.inverse() * optical_from_camera);
  return world_from_camera;
}

void SetCameraPose(open3d::visualization::Visualizer *const v,
                   const Eigen::Isometry3f &camera_pose) {
  static const Eigen::Quaternionf optical_from_camera =
      Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f::UnitX());

  // Get
  open3d::camera::PinholeCameraParameters p;
  v->GetViewControl().ConvertToPinholeCameraParameters(p);
  // Modify
  const Eigen::Isometry3f &world_from_camera = camera_pose;
  const Eigen::Isometry3f optical_from_world =
      optical_from_camera * world_from_camera.inverse();
  p.extrinsic_ = optical_from_world.matrix()
                     .cast<std::decay_t<decltype(*p.extrinsic_.data())> >();
  // Set
  v->GetViewControl().ConvertFromPinholeCameraParameters(p);
}

Eigen::Isometry3f Lerp(const Eigen::Isometry3f &x0, const Eigen::Isometry3f &x1,
                       const float w) {
  const Eigen::Vector3f d =
      x0.translation() + w * (x1.translation() - x0.translation());
  const Eigen::Quaternionf q =
      Eigen::Quaternionf{x0.linear()}.slerp(w, Eigen::Quaternionf{x1.linear()});
  return Eigen::Isometry3f{Eigen::Translation3f{d} * q};
}

// cho::gen::SdfPtr TestTangency() {
//   auto a = cho::gen::Sphere::Create(3.0);
//   a = cho::gen::Transformation::Create(a, Eigen::Vector3f{1.5, 3.1, 0});

//   auto b = cho::gen::Box::Create(Eigen::Vector3f{2.0, 1.0, 4.6});
//   b = cho::gen::Transformation::Create(b, Eigen::Vector3f{-4.5, -2.2, 0});

//   // Define Sampling domain
//   Eigen::AlignedBox3f box;
//   box.extend(Eigen::Vector3f{a->Center().array() - a->Radius()});
//   box.extend(Eigen::Vector3f{a->Center().array() + a->Radius()});
//   box.extend(Eigen::Vector3f{b->Center().array() - b->Radius()});
//   box.extend(Eigen::Vector3f{b->Center().array() + b->Radius()});

//   const int resolution{16};

//   Eigen::Array3f step{box.diagonal().array() / resolution};
//   for (int i = 0; i < resolution; ++i) {
//     for (int j = 0; j < resolution; ++j) {
//       for (int k = 0; k < resolution; ++k) {
//         const Eigen::Vector3f p{box.min().array() +
//                                 (step * Eigen::Array3f{i, j, k})};
//         a->Distance(p) + b->Distance(p);
//       }
//     }
//   }
// }

int main() {
  // Configure ...
  static constexpr const float kDegree{M_PI / 180.0F};
  static constexpr const int kVerticalResolution{128};
  static constexpr const int kHorizontalResolution{128};
  static constexpr const float kVerticalFov{120 * kDegree};
  static constexpr const float kHorizontalFov{210 * kDegree};
  static constexpr const int kNumObjects{8};
  static constexpr const bool kShow{true};
  static constexpr const bool kFollowTrajectory{true};
  static constexpr const bool kShowTrajectory{true};
  static constexpr const bool kUseFancyGui{true};

  // Initialize RNG.
  const std::int64_t seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  // const std::int64_t seed = 0;
  // const std::int64_t seed = 1609268978845759618;
  // const std::int64_t seed = 1609529784716097929;
  // const std::int64_t seed = 1609539747653922532;
  // const std::int64_t seed = 1609540336680865385;
  // const std::int64_t seed = 1609544564348054667;
  fmt::print("seed={}\n", seed);
  rng.seed(seed);

  // Create camera.
  Eigen::Vector2i resolution{kVerticalResolution, kHorizontalResolution};
  Eigen::Vector2f fov{kVerticalFov, kHorizontalFov};
  Eigen::Vector3f eye{0, 0, 0};
  Eigen::AngleAxisf eye_rot{0 * kDegree, Eigen::Vector3f::UnitY()};
  Eigen::Isometry3f eye_pose{Eigen::Translation3f{eye} * eye_rot};

  // Create scene - take in camera parameter to avoid collision.
  const cho::gen::SceneOptions scene_opts{{1, 5, 5.0, 15.0}, {1, 5}, 5, 16};
  fmt::print("Scene Generation Start.\n");
  cho::gen::SdfPtr objs;   // used for trajectory gen + debugging
  cho::gen::SdfPtr space;  // used for trajectory generation
  cho::gen::SdfPtr wall;   // unused?
  auto scene_sdf =
      cho::gen::GenerateScene(rng, eye, scene_opts, &objs, &space, &wall);
  fmt::print("Scene Generation End.\n");

  // Create trajectory.
  fmt::print("Trajectory Generation Start.\n");
  cho::gen::SdfPtr free_space = cho::gen::Subtraction::Create(space, objs);
  std::vector<Eigen::Isometry3f> trajectory;
  TrajectoryGenerationOptions traj_gen_opts{128, 32, 32.0F, 2.0F, 20 * kDegree};
  auto tt0 = std::chrono::high_resolution_clock::now();
  const bool traj_gen_suc =
      GenerateTrajectory(free_space, traj_gen_opts, &trajectory);
  auto tt1 = std::chrono::high_resolution_clock::now();
  fmt::print(
      "Gen took {} ms\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(tt1 - tt0).count());
  fmt::print("Trajectory Generation End : {}.\n",
             traj_gen_suc ? "success" : "failed");
  if (!traj_gen_suc) {
    fmt::print("ABORT!\n");
    return 1;
  }

  // NOTE(yycho0108): kShowTrajectory is disabled if following trajectory,
  // since the primitives generated for showing the trajectory would
  // occlude the camera.
  if (!kFollowTrajectory && kShowTrajectory) {
    scene_sdf = cho::gen::Union::Create(scene_sdf, TrajectoryToSdf(trajectory));
  }

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
  for (int k = 0; k < 1; ++k) {
    eye_pose_q = eye_pose_q * q;
    CreateDepthImageCuda(eye_pose_q, resolution, fov, program, &depth_image_cu,
                         &cloud_f_cu);
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  if (kUseFancyGui) {
    SdfDepthImageRendererCu depth_renderer{program, resolution, fov};

    open3d::visualization::VisualizerWithKeyCallback vis;

    // Key callbacks ...
    vis.RegisterKeyCallback(
        GLFW_KEY_UP,
        [&free_space](open3d::visualization::Visualizer *v) -> bool {
          const Eigen::Isometry3f p = GetCameraPose(v);
          // Disallow movement into colliding region.
          if (free_space->Distance(p.translation() +
                                   Eigen::Vector3f{0.1F, 0.0, 0.0}) > 0) {
            return false;
          }
          SetCameraPose(v, p * Eigen::Translation3f{0.1F, 0.0, 0.0});
          return false;
        });
    vis.RegisterKeyCallback(
        GLFW_KEY_DOWN, [](open3d::visualization::Visualizer *v) -> bool {
          const Eigen::Isometry3f p = GetCameraPose(v);
          SetCameraPose(v, p * Eigen::Translation3f{-0.1F, 0.0, 0.0});
          return false;
        });
    vis.RegisterKeyCallback(
        GLFW_KEY_LEFT, [](open3d::visualization::Visualizer *v) -> bool {
          const Eigen::Isometry3f p = GetCameraPose(v);
          SetCameraPose(v, p * Eigen::Translation3f{0.0F, 0.1F, 0.0});
          return false;
        });
    vis.RegisterKeyCallback(
        GLFW_KEY_RIGHT, [](open3d::visualization::Visualizer *v) -> bool {
          const Eigen::Isometry3f p = GetCameraPose(v);
          SetCameraPose(v, p * Eigen::Translation3f{0.0F, -0.1F, 0.0});
          return false;
        });
    vis.RegisterKeyActionCallback(
        GLFW_KEY_SPACE,
        [](open3d::visualization::Visualizer *v, const int action,
           const int mods) -> bool {
          if (action != GLFW_RELEASE) {
            return false;
          }
          if (mods & GLFW_MOD_SHIFT) {
            const Eigen::Isometry3f p = GetCameraPose(v);
            SetCameraPose(v, p * Eigen::Translation3f{0.0F, 0.0F, -0.1F});
            return false;
          } else {
            const Eigen::Isometry3f p = GetCameraPose(v);
            SetCameraPose(v, p * Eigen::Translation3f{0.0F, 0.0F, 0.1F});
            return false;
          }
          return false;
        });

    vis.CreateVisualizerWindow();

    auto cloud = O3dCloudFromVecVector3f(cloud_f_cu);
    vis.AddGeometry(cloud);

    // axes
    auto axes = open3d::geometry::TriangleMesh::CreateCoordinateFrame(
        0.5, eye.cast<double>());
    // vis.AddGeometry(axes);

    Eigen::Isometry3f eye_pose_q{eye_pose};
    Eigen::AngleAxisf q{0.1, Eigen::Vector3f::UnitZ()};

    vis.GetViewControl().SetConstantZFar(100.0F);
    vis.GetViewControl().SetConstantZNear(0.001F);
    // vis.GetViewControl().SetZoom(1.0);
    SetCameraPose(&vis, Eigen::Isometry3f{Eigen::Translation3f{eye}});

    {
      open3d::camera::PinholeCameraParameters p;
      vis.GetViewControl().ConvertToPinholeCameraParameters(p);

      // Why the hell is this interface so stupid?
      vis.GetViewControl().ChangeFieldOfView(
          (80.0F - vis.GetViewControl().GetFieldOfView()) /
          vis.GetViewControl().FIELD_OF_VIEW_STEP);
    }

    Eigen::Quaternionf q_axes = Eigen::Quaternionf::Identity();
    float tsum = 0.0;
    int tcount{0};
    while (true) {
      // Estimate FPS.
      auto t0 = std::chrono::high_resolution_clock::now();
      if (tcount > 0) {
        const float tmean = tsum / tcount / 1e6;
        const float fps = 1.0 / tmean;
        fmt::print("\rfps = {}", fps);
      }

      // Follow a linearly interpolated trajectory.
      if (kFollowTrajectory && !trajectory.empty()) {
        static constexpr const int kTimePerWaypointMs{200};
        const std::int64_t t =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();
        const int prev_trajectory_index =
            (t / kTimePerWaypointMs) % trajectory.size();
        const float alpha =
            (t % kTimePerWaypointMs) / static_cast<float>(kTimePerWaypointMs);
        const int next_trajectory_index = prev_trajectory_index + 1;
        if (next_trajectory_index >= trajectory.size()) {
          SetCameraPose(&vis, trajectory.at(prev_trajectory_index));
        } else {
          const Eigen::Isometry3f pose =
              Lerp(trajectory.at(prev_trajectory_index),
                   trajectory.at(next_trajectory_index), alpha);
          SetCameraPose(&vis, pose);
        }
      }

      eye_pose_q = GetCameraPose(&vis);
      fov.array() = static_cast<float>(
          1.0 * vis.GetViewControl().GetFieldOfView() * kDegree);

#if 1
      // CreateDepthImageCuda(eye_pose_q, resolution, fov, program,
      //                     &depth_image_cu, &cloud_f_cu);
      depth_renderer.SetFov(fov);
      depth_renderer.Render(eye_pose_q, &depth_image_cu, &cloud_f_cu);
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
      if (!vis.PollEvents()) {
        break;
      }
      vis.UpdateRender();

      auto t1 = std::chrono::high_resolution_clock::now();

      tsum += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
      ++tcount;
    }
    return 0;
  }

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

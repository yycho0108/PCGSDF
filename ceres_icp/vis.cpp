
#include <algorithm>
#include <chrono>
#include <iterator>
#include <open3d/geometry/TriangleMesh.h>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <open3d/Open3D.h>
// #include <open3d/visualization/visualizer/Visualizer.h>
#include <open3d/visualization/visualizer/GuiVisualizer.h>

#include "fmt/printf.h"

namespace o3d = open3d;

// void Draw()
// {
//     // o3d::visualization::Visualizer visualizer;
//     o3d::visualization::Visualizer visualizer;
//     if (!visualizer.CreateVisualizerWindow("window", 1024, 512, 0,
//             0)) {
//         utility::LogWarning(
//             "[DrawGeometries] Failed creating OpenGL "
//             "window.");
//         return false;
//     }
//     visualizer.GetRenderOption().point_show_normal_ = false;
//     visualizer.GetRenderOption().mesh_show_wireframe_ = false;
//     visualizer.GetRenderOption().mesh_show_back_face_ = false;
//     for (const auto& geometry_ptr : geometry_ptrs) {
//         if (!visualizer.AddGeometry(geometry_ptr)) {
//             utility::LogWarning("[DrawGeometries] Failed adding geometry.");
//             utility::LogWarning(
//                 "[DrawGeometries] Possibly due to bad geometry or wrong"
//                 " geometry type.");
//             return false;
//         }
//         // visualizer.UpdateGeometry();
//     }

//     o3d::visualization::ViewControl& view_control = visualizer.GetViewControl();
//     if (lookat != nullptr) {
//         view_control.SetLookat(*lookat);
//     }
//     if (up != nullptr) {
//         view_control.SetUp(*up);
//     }
//     if (front != nullptr) {
//         view_control.SetFront(*front);
//     }
//     if (zoom != nullptr) {
//         view_control.SetZoom(*zoom);
//     }

//     visualizer.Run();
//     visualizer.DestroyVisualizerWindow();
//     return true;
// }

int main(int argc, char* argv[])
{
    std::vector<Eigen::Vector3d> points;
    std::uniform_real_distribution<float> udist{ -1.0, 1.0 };
    std::default_random_engine rng;
    std::generate_n(std::back_inserter(points), 4096,
        [&rng, &udist]() { return Eigen::Vector3d{ udist(rng), udist(rng), udist(rng) }; });
    auto cloud_ptr = std::make_shared<o3d::geometry::PointCloud>(points);

#if 1
    auto out = cloud_ptr->HiddenPointRemoval(Eigen::Vector3d{ -5, 0, 0 }, 100);
    cloud_ptr = cloud_ptr->SelectByIndex(std::get<1>(out), false);
#endif

    //if (o3d::io::ReadPointCloud(argv[2], *cloud_ptr)) {
    //    o3d::utility::LogInfo("Successfully read {}", argv[2]);
    //} else {
    //    o3d::utility::LogWarning("Failed to read {}", argv[2]);
    //    return 1;
    //}
    // cloud_ptr->NormalizeNormals();
    fmt::print("Rendering\n");

    auto axes = o3d::geometry::TriangleMesh::CreateCoordinateFrame(0.5, Eigen::Vector3d{ -5, 0, 0 });
    o3d::visualization::DrawGeometries({ cloud_ptr, axes }, "PointCloud", 1600, 900);
    fmt::print("Rendered\n");
    return 0;
}

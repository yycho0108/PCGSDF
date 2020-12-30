#include <filament/Engine.h>

#include <open3d/visualization/gui/Button.h>
#include <open3d/visualization/rendering/filament/FilamentEngine.h>
#include <open3d/visualization/rendering/filament/FilamentResourceManager.h>
#include <open3d/visualization/visualizer/O3DVisualizer.h>

int main() {
  open3d::visualization::rendering::EngineInstance::SetResourcePath(
      "/usr/local/bin/Open3D/resources/");
  open3d::visualization::visualizer::O3DVisualizer v{"v", 640, 480};
  auto b = std::make_shared<open3d::visualization::gui::Button>("b");
  v.AddChild(b);
  return 0;
}

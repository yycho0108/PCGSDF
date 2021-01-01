
#include "cho/gen/sdf.hpp"

namespace cho {
namespace gen {

template <typename V>
inline V Pop(std::stack<V>& s) {
  V out = std::move(s.top());
  s.pop();
  return out;
}

float EvaluateSdf(const std::vector<SdfData>& program,
                  const Eigen::Vector3f& point) {
  std::stack<float> s;
  Eigen::Vector3f p{point};
  for (const auto& op : program) {
    switch (op.code) {
      case SdfOpCode::SPHERE: {
        const float radius = op.param[0];
        s.push(p.norm() - radius);
        break;
      }
      case SdfOpCode::BOX: {
        const float rx = op.param[0];
        const float ry = op.param[1];
        const float rz = op.param[2];
        const Eigen::Vector3f q = p.cwiseAbs() - Eigen::Vector3f{rx, ry, rz};
        s.push(q.cwiseMax(0).norm() + std::min(0.0F, q.maxCoeff()));
        break;
      }
      case SdfOpCode::CYLINDER: {
        const float radius = op.param[0];
        const float height = op.param[1];
        const Eigen::Vector2f d{p.head<2>().norm() - radius,
                                std::abs(p.z()) - height};
        s.push(std::min(d.maxCoeff(), 0.0F) + (d.cwiseMax(0.0F)).norm());
        break;
      }
      case SdfOpCode::PLANE: {
        const float nx = op.param[0];
        const float ny = op.param[1];
        const float nz = op.param[2];
        const float d = op.param[3];
        s.push(Eigen::Vector3f{nx, ny, nz}.dot(p) + d);
        break;
      }
      case SdfOpCode::ROUND: {
        const float d = Pop(s);
        const float r = op.param[0];
        s.push(d - r);
        break;
      }
      case SdfOpCode::NEGATION: {
        const float d = Pop(s);
        s.push(-d);
        break;
      }
      case SdfOpCode::UNION: {
        const float d0 = Pop(s);
        const float d1 = Pop(s);
        s.push(std::min(d0, d1));
        break;
      }
      case SdfOpCode::INTERSECTION: {
        const float d0 = Pop(s);
        const float d1 = Pop(s);
        s.push(std::max(d0, d1));
        break;
      }
      case SdfOpCode::SUBTRACTION: {
        const float d0 = Pop(s);
        const float d1 = Pop(s);
        s.push(std::max(-d0, d1));
        break;
      }
      case SdfOpCode::ONION: {
        const float d0 = Pop(s);
        const float thickness = op.param[0];
        s.push(std::abs(d0) - thickness);
        break;
      }
      case SdfOpCode::TRANSLATION: {
        const float tx = op.param[0];
        const float ty = op.param[1];
        const float tz = op.param[2];
        const Eigen::Vector3f v{tx, ty, tz};
        p += v;
        break;
      }
      case SdfOpCode::ROTATION: {
        const float qx = op.param[0];
        const float qy = op.param[1];
        const float qz = op.param[2];
        const float qw = op.param[3];
        const Eigen::Quaternionf q{qw, qx, qy, qz};
        p = q * p;
        break;
      }
      case SdfOpCode::TRANSFORMATION: {
        const float qx = op.param[0];
        const float qy = op.param[1];
        const float qz = op.param[2];
        const float qw = op.param[3];
        const float tx = op.param[4];
        const float ty = op.param[5];
        const float tz = op.param[6];
        const Eigen::Quaternionf q{qw, qx, qy, qz};
        const Eigen::Vector3f v{tx, ty, tz};
        p = (q * p + v);
        break;
      }
      case SdfOpCode::SCALE_BEGIN: {
        // modify `point` for the subtree.
        p *= op.param[0];
        break;
      }
      case SdfOpCode::SCALE_END: {
        const float d = Pop(s);
        s.push(d / op.param[0]);
        // restore `point` for the suptree.
        p *= op.param[0];
        break;
      }
    }
  }
  return s.top();
}
}  // namespace gen
}  // namespace cho

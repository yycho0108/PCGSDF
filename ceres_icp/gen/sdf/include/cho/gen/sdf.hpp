#pragma once

#include "cho/gen/sdf_fwd.hpp"

#include <fmt/printf.h>
#include <memory>
#include <stack>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "cho/gen/sdf_types.hpp"

namespace cho {
namespace gen {

/**
 * @brief  Linear interpolation.
 *
 * @tparam Value Type of the value to interpolate.
 * @param  v0    The value at w==0.
 * @param  v1    The value at w==1.
 * @param  w     The interpolation alpha parameter.
 *
 * @return       The interpolated value.
 */
template <typename Value>
inline Value Lerp(const Value& v0, const Value& v1, const float w) {
  return v0 + w * (v1 - v0);
}

/**
 * @brief Sdf Interface class.
 */
class SdfInterface {
 public:
  virtual ~SdfInterface() {}

  /**
   * @brief  Compute (exterior) distance to the point.
   *
   * @param  point The point to which to compute the distance.
   *
   * @return       distance to the point.
   */
  virtual float Distance(const Eigen::Vector3f& point) const = 0;
  virtual Eigen::Vector3f Center() const = 0;
  virtual float Radius() const = 0;
  virtual void Compile(std::vector<SdfData>* const program) const = 0;
  virtual std::string Jit(const std::string& point, const std::string& prefix,
                          int* const count, std::string* const subex) const = 0;
};

template <typename Derived>
class SdfBase : public SdfInterface {
 public:
  virtual inline float Distance(const Eigen::Vector3f& point) const override {
    return static_cast<const Derived*>(this)->Distance_(point);
  }
  virtual inline Eigen::Vector3f Center() const override {
    return static_cast<const Derived*>(this)->Center_();
  }
  virtual inline float Radius() const override {
    return static_cast<const Derived*>(this)->Radius_();
  }
  virtual inline void Compile(
      std::vector<SdfData>* const program) const override {
    return static_cast<const Derived*>(this)->Compile_(program);
  }
  virtual inline std::string Jit(const std::string& point,
                                 const std::string& prefix, int* const count,
                                 std::string* const subex) const override {
    return static_cast<const Derived*>(this)->Jit_(point, prefix, count, subex);
  }

  template <typename... Args>
  static SdfPtr Create(Args&&... args) {
    return std::make_shared<Derived>(args...);
  }
};

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

class Sphere : public SdfBase<Sphere> {
 public:
  explicit Sphere(const float radius) : radius_{radius} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return point.norm() - radius_;
  }
  Eigen::Vector3f Center_() const { return Eigen::Vector3f::Zero(); }
  float Radius_() const { return radius_; }

  static SdfPtr CreateFromArray(const std::array<float, 1>& data) {
    return std::make_shared<Sphere>(std::abs(data.at(0)));
  }

  void Compile_(std::vector<SdfData>* const program) const {
    program->emplace_back(
        SdfData{SdfOpCode::SPHERE, {radius_}});  // then op code
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex +=
        fmt::format("const float {} = length({}) - {};", ex, point, radius_);
    return ex;
  }

 private:
  float radius_;
};

class Box : public SdfBase<Box> {
 public:
  explicit Box(const Eigen::Vector3f& radius) : radius_{radius} {}
  float Distance_(const Eigen::Vector3f& point) const {
    const Eigen::Vector3f q = point.cwiseAbs() - radius_;
    return q.cwiseMax(0).norm() + std::min(0.0F, q.maxCoeff());
  }
  Eigen::Vector3f Center_() const { return Eigen::Vector3f::Zero(); }
  float Radius_() const { return radius_.norm(); }

  static SdfPtr CreateFromArray(const std::array<float, 3>& data) {
    return std::make_shared<Box>(
        Eigen::Vector3f{data[0], data[1], data[2]}.cwiseAbs());
  }
  void Compile_(std::vector<SdfData>* const program) const {
    // Push param in reverse order
    program->emplace_back(
        SdfData{SdfOpCode::BOX,
                {radius_.x(), radius_.y(), radius_.z()}});  // then op code
  }
  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string q = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float3 {} = abs({}) - make_float3({},{},{});",
                          q, point, radius_.x(), radius_.y(), radius_.z());
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format(
        "const float {} = length(max({},0)) + min(0, max({}));", ex, q, q);
    return ex;
  }

 private:
  Eigen::Vector3f radius_;
};

class Cylinder : public SdfBase<Cylinder> {
 public:
  explicit Cylinder(const float height, const float radius)
      : height_{height}, radius_{radius} {}
  float Distance_(const Eigen::Vector3f& point) const {
    const Eigen::Vector2f d{point.head<2>().norm() - radius_,
                            std::abs(point.z()) - height_};
    return std::min(d.maxCoeff(), 0.0F) + (d.cwiseMax(0.0F)).norm();
  }
  Eigen::Vector3f Center_() const { return Eigen::Vector3f::Zero(); }
  float Radius_() const {
    return std::sqrt(radius_ * radius_ + height_ * height_);
  }
  void Compile_(std::vector<SdfData>* const program) const {
    program->emplace_back(
        SdfData{SdfOpCode::CYLINDER, {radius_, height_}});  // then op code
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string d = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format(
        "const float2 {} = "
        "make_float2(length(make_float2({}))-{},abs({}.z)-{});",
        d, point, radius_, point, height_);
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format(
        "const float {} = min(max({}),0.0F) + length(max({},0));", ex, d, d);
    return ex;
  }

  static SdfPtr CreateFromArray(const std::array<float, 2>& data) {
    return std::make_shared<Cylinder>(std::abs(data[0]), std::abs(data[1]));
  }

 private:
  float height_;
  float radius_;
};
class Plane : public SdfBase<Plane> {
 public:
  explicit Plane(const Eigen::Vector3f& normal, const float distance)
      : normal_{normal}, distance_{distance} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return normal_.dot(point) + distance_;
  }
  Eigen::Vector3f Center_() const { return Eigen::Vector3f::Zero(); }
  float Radius_() const { return std::numeric_limits<float>::infinity(); }

  void Compile_(std::vector<SdfData>* const program) const {
    program->emplace_back(SdfData{
        SdfOpCode::PLANE, {normal_.x(), normal_.y(), normal_.z(), distance_}});
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = dot(make_float3({},{},{}),{})+ {};",
                          ex, normal_.x(), normal_.y(), normal_.z(), distance_);
    return ex;
  }

  static SdfPtr CreateFromArray(const std::array<float, 3>& data) {
    const Eigen::Vector3f v{data[0], data[1], data[2]};
    return std::make_shared<Plane>(v.normalized(), v.norm());
  }

 private:
  Eigen::Vector3f normal_;
  float distance_;
};

class Round : public SdfBase<Round> {
 public:
  explicit Round(const SdfPtr& source, const float radius)
      : source_{source}, radius_{radius} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return source_->Distance(point) - radius_;
  }
  Eigen::Vector3f Center_() const { return source_->Center(); }
  float Radius_() const { return source_->Radius(); }

  void Compile_(std::vector<SdfData>* const program) const {
    program->emplace_back(SdfData{SdfOpCode::ROUND, {radius_}});
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = {}-{};", ex,
                          source_->Jit(point, prefix, count, subex), radius_);
    return ex;
  }

 private:
  SdfPtr source_;
  float radius_;
};

class Negation : public SdfBase<Negation> {
 public:
  explicit Negation(const SdfPtr& source) : source_{source} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return -source_->Distance(point);
  }

  Eigen::Vector3f Center_() const { return source_->Center(); }
  float Radius_() const { return std::numeric_limits<float>::infinity(); }

  void Compile_(std::vector<SdfData>* const program) const {
    source_->Compile(program);
    program->emplace_back(SdfData{SdfOpCode::NEGATION, {}});
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = -{};", ex,
                          source_->Jit(point, prefix, count, subex));
    return ex;
  }

 private:
  SdfPtr source_;
};

class Union : public SdfBase<Union> {
 public:
  explicit Union(const SdfPtr& a, const SdfPtr& b) : a_{a}, b_{b} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return std::min(a_->Distance(point), b_->Distance(point));
  }

  Eigen::Vector3f Center_() const {
    const Eigen::Vector3f v = b_->Center() - a_->Center();
    const auto d = v.norm();
    if (d < std::numeric_limits<float>::epsilon()) {
      return a_->Center();
    }

    const float lhs = a_->Radius();
    const float rhs = std::max(a_->Radius(), v.norm() + b_->Radius());
    const Eigen::Vector3f u = v / d;
    return a_->Center() + u * 0.5 * (rhs - lhs);
  }
  float Radius_() const {
    const Eigen::Vector3f v = b_->Center() - a_->Center();
    const float lhs = a_->Radius();
    const float rhs = std::max(a_->Radius(), v.norm() + b_->Radius());
    return 0.5 * (a_->Radius() + rhs);
  }
  void Compile_(std::vector<SdfData>* const program) const {
    b_->Compile(program);
    a_->Compile(program);
    program->emplace_back(SdfData{SdfOpCode::UNION, {}});
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = min({},{});", ex,
                          a_->Jit(point, prefix, count, subex),
                          b_->Jit(point, prefix, count, subex));
    return ex;
  }

 private:
  SdfPtr a_;
  SdfPtr b_;
};

class Intersection : public SdfBase<Intersection> {
 public:
  explicit Intersection(const SdfPtr& a, const SdfPtr& b) : a_{a}, b_{b} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return std::max(a_->Distance(point), b_->Distance(point));
  }
  Eigen::Vector3f Center_() const {
    // TODO(yycho0108): Implement tigheter bounds.
    const Eigen::Vector3f v = b_->Center() - a_->Center();
    const float p = v.squaredNorm();
    if (p <= std::numeric_limits<float>::epsilon()) {
      return a_->Center();
    }
    const float q =
        (a_->Radius() * a_->Radius() - b_->Radius() * b_->Radius() + p) /
        (2.0f * p);
    return a_->Center() + q * v;
  }
  float Radius_() const {
    // TODO(yycho0108): Implement tigheter bounds.
    const float ra = a_->Radius();
    const float rb = b_->Radius();
    const float d_sq = (b_->Center() - a_->Center()).squaredNorm();
    if (d_sq >= std::pow(ra + rb, 2)) {
      // Too far, no intersection
      return 0;
    }

    const float ar = a_->Radius() - (Center() - a_->Center()).norm();
    const float br = b_->Radius() - (Center() - b_->Center()).norm();
    const float cr =
        std::sqrt(4 * ra * ra * d_sq - std::pow(ra * ra + d_sq - rb * rb, 2)) /
        2 * std::sqrt(d_sq);
    return std::max<float>({ar, br, cr});
  }

  void Compile_(std::vector<SdfData>* const program) const {
    b_->Compile(program);
    a_->Compile(program);
    program->emplace_back(SdfData{SdfOpCode::INTERSECTION, {}});
  }
  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = max({},{});", ex,
                          a_->Jit(point, prefix, count, subex),
                          b_->Jit(point, prefix, count, subex));
    return ex;
  }

 private:
  SdfPtr a_;
  SdfPtr b_;
};

class Subtraction : public SdfBase<Subtraction> {
 public:
  explicit Subtraction(const SdfPtr& a, const SdfPtr& b) : a_{a}, b_{b} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return std::max(-a_->Distance(point), b_->Distance(point));
  }

  Eigen::Vector3f Center_() const {
    // TODO(yycho0108): Implement tighter bounds.
    return a_->Center();
  }
  float Radius_() const {
    // TODO(yycho0108): Implement tighter bounds.
    return a_->Radius();
  }

  void Compile_(std::vector<SdfData>* const program) const {
    b_->Compile(program);
    a_->Compile(program);
    program->emplace_back(SdfData{SdfOpCode::SUBTRACTION, {}});
  }
  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = max(-{},{});", ex,
                          a_->Jit(point, prefix, count, subex),
                          b_->Jit(point, prefix, count, subex));
    return ex;
  }

 private:
  SdfPtr a_;
  SdfPtr b_;
};

class SmoothUnion : public SdfBase<SmoothUnion> {
 public:
  explicit SmoothUnion(const SdfPtr& a, const SdfPtr& b, const float k)
      : a_{a}, b_{b}, k_{k} {}
  float Distance_(const Eigen::Vector3f& point) const {
    const float d1 = a_->Distance(point);
    const float d2 = b_->Distance(point);
    const float h =
        std::min<float>(1, std::max<float>(0, 0.5 + 0.5 * (d2 - d1) / k_));
    return Lerp(d2, d1, h) - k_ * h * (1.0 - h);
  }

  Eigen::Vector3f Center_() const {
    const Eigen::Vector3f v = b_->Center() - a_->Center();
    const auto d = v.norm();
    if (d < std::numeric_limits<float>::epsilon()) {
      return a_->Center();
    }
    const Eigen::Vector3f u = v / d;
    return a_->Center() + u * (0.5 * d + b_->Radius());
  }
  float Radius_() const {
    const Eigen::Vector3f v = b_->Center() - a_->Center();
    return a_->Radius() + v.norm() + b_->Radius();
  }

  void Compile_(std::vector<SdfData>* const program) const {
    b_->Compile(program);
    a_->Compile(program);
    // TODO(yycho0108): implement UNION_S.
    program->emplace_back(SdfData{SdfOpCode::UNION, {}});
  }
  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    // TODO(yycho0108): implement UNION_S.
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = min({},{});", ex,
                          a_->Jit(point, prefix, count, subex),
                          b_->Jit(point, prefix, count, subex));
    return ex;
  }

 private:
  SdfPtr a_;
  SdfPtr b_;
  float k_;
};

class Onion : public SdfBase<Onion> {
 public:
  explicit Onion(const SdfPtr& source, const float thickness)
      : source_{source}, thickness_{thickness} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return std::abs(source_->Distance(point)) - thickness_;
  }

  Eigen::Vector3f Center_() const {
    // TODO(yycho0108): Implement tighter bounds.
    return source_->Center();
  }
  float Radius_() const {
    // TODO(yycho0108): Implement tighter bounds.
    return source_->Radius() + thickness_;
  }
  void Compile_(std::vector<SdfData>* const program) const {
    source_->Compile(program);
    program->emplace_back(SdfData{SdfOpCode::ONION, {thickness_}});
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex +=
        fmt::format("const float {} = abs({}) - {};", ex,
                    source_->Jit(point, prefix, count, subex), thickness_);
    return ex;
  }

 private:
  SdfPtr source_;
  float thickness_;
};

class Transformation : public SdfBase<Transformation> {
 public:
  explicit Transformation(const SdfPtr& source, const Eigen::Isometry3f& xfm)
      : source_{source}, xfm_{xfm}, xfm_inv_{xfm.inverse()} {}
  explicit Transformation(const SdfPtr& source, const Eigen::Quaternionf& xfm)
      : source_{source}, xfm_{xfm}, xfm_inv_{xfm.inverse()} {}
  explicit Transformation(const SdfPtr& source, const Eigen::Translation3f& xfm)
      : source_{source}, xfm_{xfm}, xfm_inv_{xfm.inverse()} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return source_->Distance(xfm_inv_ * point);
  }
  Eigen::Vector3f Center_() const {
    return source_->Center() + xfm_.translation();
  }
  float Radius_() const { return source_->Radius(); }

  void Compile_(std::vector<SdfData>* const program) const {
    const auto& ti = xfm_inv_.translation();
    const Eigen::Quaternionf qi{xfm_inv_.linear()};
    const auto& t = xfm_.translation();
    const Eigen::Quaternionf q{xfm_.linear()};

    // Apply inverse transform to point.
    program->emplace_back(
        SdfData{SdfOpCode::TRANSFORMATION,
                {qi.x(), qi.y(), qi.z(), qi.w(), ti.x(), ti.y(), ti.z()}});

    source_->Compile(program);

    // Apply forward transform to point.
    program->emplace_back(
        SdfData{SdfOpCode::TRANSFORMATION,
                {q.x(), q.y(), q.z(), q.w(), t.x(), t.y(), t.z()}});
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string tpoint = fmt::format("{}{}", prefix, (*count)++);

    const auto& ti = xfm_inv_.translation();
    const Eigen::Quaternionf qi{xfm_inv_.linear()};

    *subex += fmt::format(
        "const float3 {} = rotate(make_float4({},{},{},{}), {}) + "
        "make_float3({},{},{});",
        tpoint, qi.x(), qi.y(), qi.z(), qi.w(), point, ti.x(), ti.y(), ti.z());

#if 0
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = {};", ex,
                          source_->Jit(tpoint, prefix, count, subex));
    return ex;
#else
    return source_->Jit(tpoint, prefix, count, subex);
#endif
  }

 private:
  SdfPtr source_;
  Eigen::Isometry3f xfm_;
  Eigen::Isometry3f xfm_inv_;
};

class Scale : public SdfBase<Scale> {
 public:
  explicit Scale(const SdfPtr& source, const float scale)
      : source_{source}, scale_{scale}, scale_inv_{1.0F / scale} {}
  float Distance_(const Eigen::Vector3f& point) const {
    return source_->Distance(point * scale_inv_) * scale_;
  }
  Eigen::Vector3f Center_() const { return source_->Center(); }
  float Radius_() const { return scale_ * source_->Radius(); }
  void Compile_(std::vector<SdfData>* const program) const {
    program->emplace_back(SdfData{SdfOpCode::SCALE_BEGIN, {scale_inv_}});
    source_->Compile(program);
    program->emplace_back(SdfData{SdfOpCode::SCALE_END, {scale_}});
  }

  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    const std::string spoint = fmt::format("{}{}", prefix, (*count)++);
    *subex +=
        fmt::format("const float3 {} = {} * {};", spoint, scale_inv_, point);

    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = {} * {};", ex, scale_,
                          source_->Jit(spoint, prefix, count, subex));
    return ex;
  }

 private:
  SdfPtr source_;
  float scale_;
  float scale_inv_;
};

template <typename Sdf>
class traits {};

template <>
class traits<Sphere> {
 public:
  static constexpr int DoF = 1;
};

template <>
class traits<Box> {
 public:
  static constexpr int DoF = 3;
};

template <>
class traits<Plane> {
 public:
  static constexpr int DoF = 3;
};

template <>
class traits<Cylinder> {
 public:
  static constexpr int DoF = 2;
};

template <>
class traits<Round> {
 public:
  static constexpr int DoF = 1;
};

template <>
class traits<Negation> {
 public:
  static constexpr int DoF = 0;
};

template <>
class traits<Union> {
 public:
  static constexpr int DoF = 0;
};

template <>
class traits<Intersection> {
 public:
  static constexpr int DoF = 0;
};

template <>
class traits<Subtraction> {
 public:
  static constexpr int DoF = 0;
};

template <>
class traits<Transformation> {
 public:
  static constexpr int DoF = 6;
};

template <>
class traits<Scale> {
 public:
  static constexpr int DoF = 1;
};
}  // namespace gen
}  // namespace cho

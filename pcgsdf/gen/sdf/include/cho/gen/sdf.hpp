#pragma once

#include "cho/gen/sdf_fwd.hpp"

#include <fmt/printf.h>
#include <iostream>
#include <memory>
#include <stack>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "cho/gen/sdf_types.hpp"

#define USE_JIT_TEMP 1

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

float EvaluateSdf(const std::vector<SdfData>& program,
                  const Eigen::Vector3f& point);

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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex +=
        fmt::format("const float {} = length({}) - {};", ex, point, radius_);
    return ex;
#else
    return fmt::format("(length({}) - {})", point, radius_);
#endif
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format(
        "const float {} = length(max({},0)) + min(0, max({}));", ex, q, q);
    return ex;
#else
    return fmt::format("(length(max({},0)) + min(0, max({})))", q, q);
#endif
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format(
        "const float {} = min(max({}),0.0F) + length(max({},0));", ex, d, d);
    return ex;
#else
    return fmt::format("(min(max({}),0.0F) + length(max({},0)))", d, d);
#endif
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = dot(make_float3({},{},{}),{})+ {};",
                          ex, normal_.x(), normal_.y(), normal_.z(), distance_);
    return ex;
#else
    return fmt::format("(dot(make_float3({},{},{}),{}) + {})", normal_.x(),
                       normal_.y(), normal_.z(), distance_);
#endif
  }

  static SdfPtr CreateFromArray(const std::array<float, 3>& data) {
    const Eigen::Vector3f v{data[0], data[1], data[2]};
    return std::make_shared<Plane>(v.normalized(), v.norm());
  }

 private:
  Eigen::Vector3f normal_;
  float distance_;
};

class Torus : public SdfBase<Torus> {
 public:
  explicit Torus(const float major_radius, const float minor_radius)
      : r0_{major_radius}, r1_{minor_radius} {}
  float Distance_(const Eigen::Vector3f& point) const {
    const Eigen::Vector2f q{point.head<2>().norm() - r0_, point.z()};
    return q.norm() - r1_;
  }
  Eigen::Vector3f Center_() const { return Eigen::Vector3f::Zero(); }
  float Radius_() const { return r0_ + r1_; }
  void Compile_(std::vector<SdfData>* const program) const {
    program->emplace_back(SdfData{SdfOpCode::TORUS, {r0_, r1_}});
  }
  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    throw std::runtime_error("torus jit not implemented");
    return "";
  }
  static SdfPtr CreateFromArray(const std::array<float, 2>& data) {
    return std::make_shared<Torus>(std::abs(data[0]), std::abs(data[1]));
  }

 private:
  float r0_;
  float r1_;
};

inline float Cross2(const Eigen::Vector2f& lhs, const Eigen::Vector2f& rhs) {
  return lhs.x() * rhs.y() - lhs.y() * rhs.x();
}

class Cone : public SdfBase<Cone> {
 public:
  explicit Cone(const float radius, const float height)
      : radius_(radius), height_(height) {}
  float Distance_(const Eigen::Vector3f& point) const {
    const Eigen::Vector2f q{radius_, height_};
    const Eigen::Vector2f w{point.head<2>().norm(), point.z()};

    const Eigen::Vector2f d{w - q};
    const float k = q.y() / q.x();
    const float k2 = q.x() * q.y() / (q.x() + q.norm());
    const Eigen::Vector2f pr{q.x(), 0};
    const Eigen::Vector2f ph{0, q.y()};
    // (-ph.y)*w.x - (pr.x)*w.y
    return k * d.y() >= w.x()
               ? (w - ph).norm()
               : ((w.y() >= -k2 * d.x() && k * w.y() >= d.x())
                      ? Cross2(q, w - ph)
                      : (w.x() >= q.x() ? (w - pr).norm() : -w.y()));
  }
  // TODO(ycho): Implement
  Eigen::Vector3f Center_() const { return Eigen::Vector3f::Zero(); }
  float Radius_() const { return radius_ * radius_ + height_ * height_; }

  void Compile_(std::vector<SdfData>* const program) const {
    program->emplace_back(SdfData{SdfOpCode::CONE, {radius_, height_}});
  }
  std::string Jit_(const std::string& point, const std::string& prefix,
                   int* const count, std::string* const subex) const {
    throw std::runtime_error("cone jit not implemented");
    return "";
  }
  static SdfPtr CreateFromArray(const std::array<float, 2>& data) {
    return std::make_shared<Cone>(std::abs(data[0]), std::abs(data[1]));
  }

 private:
  float radius_;
  float height_;
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = {}-{};", ex,
                          source_->Jit(point, prefix, count, subex), radius_);
    return ex;
#else
    return fmt::format("({}-{})", source_->Jit(point, prefix, count, subex),
                       radius_);
#endif
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = -{};", ex,
                          source_->Jit(point, prefix, count, subex));
    return ex;
#else
    return fmt::format("(-{})", source_->Jit(point, prefix, count, subex));
#endif
  }

 private:
  SdfPtr source_;
};

class OpUnion : public SdfBase<OpUnion> {
 public:
  explicit OpUnion(const SdfPtr& a, const SdfPtr& b) : a_{a}, b_{b} {}
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = min({},{});", ex,
                          a_->Jit(point, prefix, count, subex),
                          b_->Jit(point, prefix, count, subex));
    return ex;
#else
    return fmt::format("min({},{})", a_->Jit(point, prefix, count, subex),
                       b_->Jit(point, prefix, count, subex));
#endif
  }

 private:
  SdfPtr a_;
  SdfPtr b_;
};

class OpIntersection : public SdfBase<OpIntersection> {
 public:
  explicit OpIntersection(const SdfPtr& a, const SdfPtr& b) : a_{a}, b_{b} {}
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = max({},{});", ex,
                          a_->Jit(point, prefix, count, subex),
                          b_->Jit(point, prefix, count, subex));
    return ex;
#else
    return fmt::format("max({},{})", a_->Jit(point, prefix, count, subex),
                       b_->Jit(point, prefix, count, subex));
#endif
  }

 private:
  SdfPtr a_;
  SdfPtr b_;
};

class OpSubtraction : public SdfBase<OpSubtraction> {
 public:
  explicit OpSubtraction(const SdfPtr& a, const SdfPtr& b) : a_{a}, b_{b} {}
  float Distance_(const Eigen::Vector3f& point) const {
    // a - b == Intersection(a, Negation(b))
    return std::max(a_->Distance(point), -b_->Distance(point));
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = max(-{},{});", ex,
                          a_->Jit(point, prefix, count, subex),
                          b_->Jit(point, prefix, count, subex));
    return ex;
#else
    return fmt::format("max(-{},{})", a_->Jit(point, prefix, count, subex),
                       b_->Jit(point, prefix, count, subex));
#endif
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex += fmt::format("const float {} = min({},{});", ex,
                          a_->Jit(point, prefix, count, subex),
                          b_->Jit(point, prefix, count, subex));
    return ex;
#else
    return fmt::format("min({},{})", a_->Jit(point, prefix, count, subex),
                       b_->Jit(point, prefix, count, subex));
#endif
  }

 private:
  SdfPtr a_;
  SdfPtr b_;
  float k_;
};

class OpOnion : public SdfBase<OpOnion> {
 public:
  explicit OpOnion(const SdfPtr& source, const float thickness)
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
#if USE_JIT_TEMP
    const std::string ex = fmt::format("{}{}", prefix, (*count)++);
    *subex +=
        fmt::format("const float {} = abs({}) - {};", ex,
                    source_->Jit(point, prefix, count, subex), thickness_);
    return ex;
#else
    return fmt::format("(abs({}) - {})",
                       source_->Jit(point, prefix, count, subex), thickness_);
#endif
  }

 private:
  SdfPtr source_;
  float thickness_;
};

class OpTransformation : public SdfBase<OpTransformation> {
 public:
  explicit OpTransformation(const SdfPtr& source, const Eigen::Isometry3f& xfm)
      : source_{source}, xfm_{xfm}, xfm_inv_{xfm.inverse()} {}
  explicit OpTransformation(const SdfPtr& source, const Eigen::Quaternionf& xfm)
      : source_{source}, xfm_{xfm}, xfm_inv_{xfm.inverse()} {}
  explicit OpTransformation(const SdfPtr& source,
                            const Eigen::Translation3f& xfm)
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
class traits<Torus> {
 public:
  static constexpr int DoF = 2;
};

template <>
class traits<Cone> {
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
class traits<OpUnion> {
 public:
  static constexpr int DoF = 0;
};

template <>
class traits<OpIntersection> {
 public:
  static constexpr int DoF = 0;
};

template <>
class traits<OpSubtraction> {
 public:
  static constexpr int DoF = 0;
};

template <>
class traits<OpTransformation> {
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

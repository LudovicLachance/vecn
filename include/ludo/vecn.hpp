#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

template <class T, class U>
constexpr bool equal(const T &lhs, const U &rhs) {
  auto value = static_cast<float>(lhs - rhs);
  return value <= 0.0001f && value >= -0.0001f;
}

template <class T, uint64_t N>
struct vecn : public std::array<T, N> {
  void zero() { this->fill(T{0}); }

  // Assign
  vecn<T, N> &operator=(vecn<T, N> rhs) {
    this->swap(rhs);
    return (*this);
  }

  // +
  vecn<T, N> &operator+=(const vecn<T, N> &rhs) {
    for (size_t i = 0; i < this->size(); ++i) {
      (*this)[i] += rhs[i];
    }
    return (*this);
  }
  friend vecn<T, N> operator+(vecn<T, N> lhs, const vecn<T, N> &rhs) {
    lhs += rhs;
    return lhs;
  }

  // -
  vecn<T, N> operator-() const {
    vecn<T, N> tmp{};
    for (size_t i = 0; i < this->size(); ++i) {
      tmp[i] = -(*this)[i];
    }
    return tmp;
  }
  vecn<T, N> &operator-=(const vecn<T, N> &rhs) {
    for (size_t i = 0; i < this->size(); ++i) {
      (*this)[i] -= rhs[i];
    }
    return (*this);
  }
  friend vecn<T, N> operator-(vecn<T, N> lhs, const vecn<T, N> &rhs) {
    lhs -= rhs;
    return lhs;
  }

  // *
  vecn<T, N> &operator*=(const vecn<T, N> &rhs) {
    for (size_t i = 0; i < this->size(); ++i) {
      (*this)[i] *= rhs[i];
    }
    return (*this);
  }
  friend vecn<T, N> operator*(vecn<T, N> lhs, const vecn<T, N> &rhs) {
    lhs *= rhs;
    return lhs;
  }
  friend vecn<T, N> operator*(vecn<T, N> lhs, const T &rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
      lhs[i] *= rhs;
    }
    return lhs;
  }

  // /
  vecn<T, N> &operator/=(const vecn<T, N> &rhs) {
    for (size_t i = 0; i < this->size(); ++i) {
      (*this)[i] /= rhs[i];
    }
    return (*this);
  }
  friend vecn<T, N> operator/(vecn<T, N> lhs, const vecn<T, N> &rhs) {
    lhs /= rhs;
    return lhs;
  }
  friend vecn<T, N> operator/(vecn<T, N> lhs, const T &rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
      lhs[i] /= rhs;
    }
    return lhs;
  }

  T dotProduct(const vecn<T, N> &rhs) const {
    vecn<T, N> tmp = ((*this) * rhs);
    return std::accumulate(std::begin(tmp), std::end(tmp), T{0},
                           std::plus<T>());
  }
  vecn<T, N> projection(const vecn<T, N> &rhs) const {
    return rhs * (this->dotProduct(rhs) / rhs.dotProduct(rhs));
  }
  vecn<T, N> crossProduct(const vecn<T, N> &rhs) const {
    vecn<T, N> tmp;
    auto size = this->size();
    for (size_t i = 0; i < size; ++i) {
      auto i1 = ((i + 1) % size);
      auto i2 = ((i + 2) % size);
      tmp[i] = (((*this)[i1] * rhs[i2]) - ((*this)[i2] * rhs[i1]));
    }
    return tmp;
  }
  vecn<T, N> diff(const vecn<T, N> &rhs) const { return rhs - (*this); }
  bool parallel(const vecn<T, N> &rhs) const {
    return equal(this->crossProduct(rhs).magnitude(), 0.f);
  }
  bool perpendicular(const vecn<T, N> &rhs) const {
    return equal(this->dotProduct(rhs), 0.f);
  }
  T magnitude() const { return std::sqrt(this->dotProduct((*this))); }
  bool unitary() const { return equal(this->magnitude(), 1.f); }
  T distance(const vecn<T, N> &rhs) const {
    return this->diff(rhs).magnitude();
  }

  friend bool operator==(const vecn<T, N> &lhs, const vecn<T, N> &rhs) {
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i] != rhs[i]) {
        return false;
      }
    }
    return true;
  }
  friend bool operator!=(const vecn<T, N> &lhs, const vecn<T, N> &rhs) {
    return !operator==(lhs, rhs);
  }
  friend bool operator<(const vecn<T, N> &lhs, const vecn<T, N> &rhs) {
    return lhs.magnitude() < rhs.magnitude();
  }
  friend bool operator>(const vecn<T, N> &lhs, const vecn<T, N> &rhs) {
    return operator<(rhs, lhs);
  }
  friend bool operator<=(const vecn<T, N> &lhs, const vecn<T, N> &rhs) {
    return !operator>(lhs, rhs);
  }
  friend bool operator>=(const vecn<T, N> &lhs, const vecn<T, N> &rhs) {
    return !operator<(lhs, rhs);
  }
};

template <class T, uint64_t N>
class matn : public vecn<vecn<T, N>, N> {
 public:
  void zero() {
    for (size_t i = 0; i < this->size(); ++i) {
      (*this)[i].zero();
    }
  }

  T determinant() const {
    auto subdet = subDeterminant();
    return std::accumulate(std::begin(subdet), std::end(subdet), T{0},
                           std::plus<T>());
  }

  vecn<T, N> subDeterminant(bool wbase = true, bool wmult = true) const {
    vecn<T, N> tmp;
    T mult{1};
    T det{0};
    for (size_t x = 0; x < tmp.size(); ++x) {
      matn<T, N - 1ull> tmpMat;
      for (size_t ry = 1; ry < N; ++ry) {
        size_t tx = 0;
        for (size_t rx = 0; rx < N; ++rx) {
          if (rx == x) {
            continue;
          }
          tmpMat[ry - 1ull][tx] = (*this)[ry][rx];
          ++tx;
        }
      }
      if (x % 2 == 0) {
        mult = T{1};
      } else {
        mult = T{-1};
      }
      T base{1};
      if (wbase) {
        base = (*this)[0][x];
      }
      if (!wmult) {
        mult = T{1};
      }
      tmp[x] = (mult * (base * tmpMat.determinant()));
    }
    return tmp;
  }

  matn<T, N> minorMat() const {
    matn<T, N> tmp;
    auto size = tmp.size();
    for (size_t y = 0; y < size; ++y) {
      matn<T, N> copy = *this;
      for (size_t i = 0; i < y; ++i) {
        std::swap(copy[i], copy[y]);
      }
      auto subD = copy.subDeterminant(false, false);
      for (size_t x = 0; x < size; ++x) {
        tmp[y][x] = subD[x];
      }
    }
    return tmp;
  }

  matn<T, N> cofactorMat() const {
    matn<T, N> tmp = *this;
    auto size = tmp.size();
    for (size_t y = 0; y < size; ++y) {
      for (size_t x = 0; x < size; ++x) {
        tmp[y][x] = pow(-1, y + x) * tmp[y][x];
      }
    }
    return tmp;
  }

  matn<T, N> transpose() const {
    matn<T, N> tmp = *this;
    auto size = this->size();
    for (size_t y = 0; y < size; ++y) {
      for (size_t x = y; x < size; ++x) {
        std::swap(tmp[y][x], tmp[x][y]);
      }
    }
    return tmp;
  }

  matn<T, N> inverse() const {
    auto det = this->determinant();
    assert(!equal(det, 0));
    return this->minorMat().cofactorMat().transpose() / det;
  }

  vecn<T, N> solve(const vecn<T, N> &rhs) {
    vecn<T, N> tmp;
    auto inverse = this->inverse();
    for (size_t i = 0; i < rhs.size(); ++i) {
      tmp[i] = inverse[i].dotProduct(rhs);
    }
    return tmp;
  }
  // *
  matn<T, N> &operator*=(const matn<T, N> &rhs) {
    matn<T, N> tmp = rhs.transpose();
    matn<T, N> copy = *this;
    auto size = this->size();
    for (size_t y = 0; y < size; ++y) {
      for (size_t x = 0; x < size; ++x) {
        (*this)[y][x] = copy[y].dotProduct(tmp[x]);
      }
    }
    return (*this);
  }
  friend matn<T, N> operator*(matn<T, N> lhs, const matn<T, N> &rhs) {
    lhs *= rhs;
    return lhs;
  }

  // /
  matn<T, N> &operator/=(const T &rhs) {
    auto size = this->size();
    for (size_t y = 0; y < size; ++y) {
      (*this)[y] = (*this)[y] / rhs;
    }
    return (*this);
  }
  friend matn<T, N> operator/(matn<T, N> lhs, const T &rhs) {
    lhs /= rhs;
    return lhs;
  }

  static auto identity() -> matn<T, N> {
    matn<T, N> tmp;
    tmp.zero();
    for (size_t i = 0; i < tmp.size(); ++i) {
      tmp[i][i] = 1;
    }
    return tmp;
  }
};

template <class T>
class matn<T, 1> : public vecn<vecn<T, 1>, 1> {
 public:
  T determinant() const { return (*this)[0][0]; }
};

template <class T>
class matn<T, 3> : public vecn<vecn<T, 3>, 3> {
 public:
  T determinant() const {
    auto subdet = subDeterminant();
    return std::accumulate(std::begin(subdet), std::end(subdet), T{0},
                           std::plus<T>());
  }

  vecn<T, 3> subDeterminant(bool wbase = true, bool wmult = true) const {
    vecn<T, 3> tmp;
    T mult{1};
    T det{0};
    for (size_t x = 0; x < tmp.size(); ++x) {
      matn<T, 2> tmpMat;
      for (size_t ry = 1; ry < 3; ++ry) {
        size_t tx = 0;
        for (size_t rx = 0; rx < 3; ++rx) {
          if (rx == x) {
            continue;
          }
          tmpMat[ry - 1ull][tx] = (*this)[ry][rx];
          ++tx;
        }
      }
      if (x % 2 == 0) {
        mult = T{1};
      } else {
        mult = T{-1};
      }
      T base{1};
      if (wbase) {
        base = (*this)[0][x];
      }
      if (!wmult) {
        mult = T{1};
      }
      tmp[x] = (mult * (base * tmpMat.determinant()));
    }
    return tmp;
  }

  matn<T, 3> minorMat() const {
    matn<T, 3> tmp;
    auto size = tmp.size();
    for (size_t y = 0; y < size; ++y) {
      matn<T, 3> copy = *this;
      for (size_t i = 0; i < y; ++i) {
        std::swap(copy[i], copy[y]);
      }
      auto subD = copy.subDeterminant(false, false);
      for (size_t x = 0; x < size; ++x) {
        tmp[y][x] = subD[x];
      }
    }
    return tmp;
  }

  matn<T, 3> cofactorMat() const {
    matn<T, 3> tmp = *this;
    auto size = tmp.size();
    for (size_t y = 0; y < size; ++y) {
      for (size_t x = 0; x < size; ++x) {
        tmp[y][x] = pow(-1, y + x) * tmp[y][x];
      }
    }
    return tmp;
  }

  matn<T, 3> transpose() const {
    matn<T, 3> tmp = *this;
    auto size = this->size();
    for (size_t y = 0; y < size; ++y) {
      for (size_t x = y; x < size; ++x) {
        std::swap(tmp[y][x], tmp[x][y]);
      }
    }
    return tmp;
  }

  matn<T, 3> inverse() const {
    auto det = this->determinant();
    assert(!equal(det, 0));
    return this->minorMat().cofactorMat().transpose() / det;
  }

  vecn<T, 3> solve(const vecn<T, 3> &rhs) {
    vecn<T, 3> tmp;
    auto inverse = this->inverse();
    for (size_t i = 0; i < rhs.size(); ++i) {
      tmp[i] = inverse[i].dotProduct(rhs);
    }
    return tmp;
  }

  // *
  matn<T, 3> &operator*=(const matn<T, 3> &rhs) {
    matn<T, 3> tmp = rhs.transpose();
    matn<T, 3> copy = *this;
    auto size = this->size();
    for (size_t y = 0; y < size; ++y) {
      for (size_t x = 0; x < size; ++x) {
        (*this)[y][x] = copy[y].dotProduct(tmp[x]);
      }
    }
    return (*this);
  }
  friend matn<T, 3> operator*(matn<T, 3> lhs, const matn<T, 3> &rhs) {
    lhs *= rhs;
    return lhs;
  }

  // /
  matn<T, 3> &operator/=(const T &rhs) {
    auto size = this->size();
    for (size_t y = 0; y < size; ++y) {
      (*this)[y] = (*this)[y] / rhs;
    }
    return (*this);
  }
  friend matn<T, 3> operator/(matn<T, 3> lhs, const T &rhs) {
    lhs /= rhs;
    return lhs;
  }

  static auto rotation(const T &rad) -> matn<T, 3> {
    return {
        vecn<T, 3>{cos(rad), sin(rad), 0},
        vecn<T, 3>{-sin(rad), cos(rad), 0},
        vecn<T, 3>{0, 0, 1},
    };
  }
  static auto rotationCWize(const T &rad) -> matn<T, 3> {
    return rotation(-rad);
  }
  static auto rotationACWize(const T &rad) -> matn<T, 3> {
    return rotation(rad);
  }
  static auto translation(const vecn<T, 2> &vec) -> matn<T, 3> {
    return {
        vecn<T, 3>{1, 0, vec[0]},
        vecn<T, 3>{0, 1, vec[1]},
        vecn<T, 3>{0, 0, 1},
    };
  }
  static auto scale(const vecn<T, 2> &vec) -> matn<T, 3> {
    return {
        vecn<T, 3>{vec[0], 0, 0},
        vecn<T, 3>{0, vec[1], 0},
        vecn<T, 3>{0, 0, 1},
    };
  }
  static auto scale(const T &value) -> matn<T, 3> {
    return scale({value, value});
  }
  static auto reflectAboutOrigin() -> matn<T, 3> { return scale({-1, -1}); }
  static auto reflectAboutXAxis() -> matn<T, 3> { return scale({1, -1}); }
  static auto reflectAboutYAxis() -> matn<T, 3> { return scale({-1, 1}); }
  static auto identity() -> matn<T, 3> {
    matn<T, 3> tmp;
    tmp.zero();
    for (size_t i = 0; i < tmp.size(); ++i) {
      tmp[i][i] = 1;
    }
    return tmp;
  }
  static auto shearInXAxis(const T &rad) -> matn<T, 3> {
    return {{1, tan(rad), 0}, {0, 1, 0}, {0, 0, 1}};
  }
  static auto shearInYAxis(const T &rad) -> matn<T, 3> {
    return {{1, 0, 0}, {tan(rad), 1, 0}, {0, 0, 1}};
  }
};

template <class T, uint64_t N>
vecn<T, N - 1> operator*=(vecn<T, N - 1> &vec, const matn<T, N> &mat) {
  vecn<T, N - 1> tmp;
  for (size_t i = 0; i < tmp.size(); ++i) {
    T sum{0};
    for (size_t vi = 0; vi < vec.size(); ++vi) {
      sum += (vec[vi] * mat[i][vi]);
    }
    tmp[i] = sum;
  }
  vec = tmp;
  return vec;
}
template <class T, uint64_t N>
vecn<T, N - 1> operator*(vecn<T, N - 1> vec, const matn<T, N> &mat) {
  vec *= mat;
  return vec;
}

template <class T, uint64_t N>
vecn<T, N> operator*=(vecn<T, N> &vec, const matn<T, N> &mat) {
  vecn<T, N> tmp;
  for (size_t i = 0; i < tmp.size(); ++i) {
    // T sum{0};
    // for (size_t vi = 0; vi < vec.size(); ++vi) {
    //   sum += (vec[vi] * mat[i][vi]);
    // }
    tmp[i] = vec.dotProduct(mat[i]);
  }
  vec = tmp;
  return vec;
}
template <class T, uint64_t N>
vecn<T, N> operator*(vecn<T, N> vec, const matn<T, N> &mat) {
  vec *= mat;
  return vec;
}

template <class T>
using vec2 = vecn<T, 2>;
template <class T>
using vec3 = vecn<T, 3>;
template <class T>
using vec4 = vecn<T, 4>;

template <class T>
using mat2 = matn<T, 2>;
template <class T>
using mat3 = matn<T, 3>;
template <class T>
using mat4 = matn<T, 4>;
#pragma once

template <typename T = nullptr_t>
class tensor::Vector {
    static constexpr auto serial_class = detail::SerialClass::VECTOR;

public:
    bool transposed = false;
    uint32_t size;
    std::vector<T> vec;

    Vector() = default;

    explicit Vector(const uint32_t size, const T& value = T()) : size(size), vec(size, value) {}

    Vector(std::initializer_list<T> list) : size(list.size()), vec(list.begin(), list.end()) {}

    template <typename U, typename t = decltype(std::declval<T>() * std::declval<U>()), typename R = decltype(std::declval<t>() + std::declval<t>())>
    R operator*(const Vector<U> other) {
        assert((transposed || other.transposed) && size == other.size);
        R res;

        for (uint32_t i = 0; i < size; i++) {
            res += vec[i] * other[i];
        }
        return res;
    }

    T& operator[](const uint32_t i) { return vec[i]; }

    const T& operator[](const uint32_t i) const { return vec[i]; }

    Vector transpose() const {
        Vector res = *this;
        res.transposed ^= true;
        return res;
    }

    template <typename U, typename t = decltype(std::declval<T>() * std::declval<U>()), typename R = decltype(std::declval<t>() + std::declval<t>())>
    R dot(const Vector<U>& value) const {
        assert(size == value.size);
        R res;

        for (uint32_t i = 0; i < size; i++) {
            res += vec[i] * value.vec[i];
        }
        return res;
    }

    template <typename U>
    static Vector<U> differentiate(const U& scalar, const Vector<algebra::Variable>& wrt, const bool origin = true) {
        Vector<U> res(wrt.size);

        if constexpr (requires(const U& obj, const algebra::Variable& variable) { obj.differentiate(variable); }) {
            for (uint32_t i = 0; i < wrt.size; i++) {
                res[i] = scalar.differentiate(wrt[i], false);
            }
        }
        res.transposed = true;

        if (origin && GLOBAL_FORMATTING.verbose) {
            algebra::detail::print_differentiate(scalar, wrt, res);
        }
        return res;
    }

    static Vector<algebra::RationalPolynomial> gradient(const algebra::RationalPolynomial& polynomial, const bool origin = true) {
        Vector<algebra::RationalPolynomial> res({polynomial.differentiate(algebra::Variable("x"), false),
                                                 polynomial.differentiate(algebra::Variable("y"), false),
                                                 polynomial.differentiate(algebra::Variable("z"), false)});

        if (origin && GLOBAL_FORMATTING.verbose) {
            if (GLOBAL_FORMATTING.output == algebra::detail::FormatSettings::Output::LATEX) {
                GLOBAL_FORMATTING << algebra::detail::LaTeX(
                    std::string("\\nabla\\left(").append(polynomial.to_latex()).append("\\right)=").append(res.to_latex()));
            } else {
                GLOBAL_FORMATTING << "grad(" << polynomial << ") = " << res;
            }
        }
        return res;
    }

    algebra::RationalPolynomial divergence(const bool origin = true) const {
        assert(size == 3);
        algebra::RationalPolynomial res;

        if constexpr (requires(const T& obj, const algebra::Variable& variable, const bool origin) { obj.differentiate(variable, origin); }) {
            res += vec[0].differentiate(algebra::Variable("x"), false);
            res += vec[1].differentiate(algebra::Variable("y"), false);
            res += vec[2].differentiate(algebra::Variable("z"), false);
        }
        if (origin && GLOBAL_FORMATTING.verbose) {
            if (GLOBAL_FORMATTING.output == algebra::detail::FormatSettings::Output::LATEX) {
                GLOBAL_FORMATTING << algebra::detail::LaTeX(std::string("\\nabla\\cdot").append(to_latex()).append("=").append(res.to_latex()));
            } else {
                GLOBAL_FORMATTING << "div(" << *this << ") = " << res;
            }
        }
        return res;
    }

    Vector curl(const bool origin = true) const {
        assert(size == 3);
        algebra::Variable x("x"), y("y"), z("z");
        Vector res(3);

        if constexpr (requires(const T& obj, const algebra::Variable& variable, const bool origin) { obj.differentiate(variable, origin); }) {
            res[0] = vec[2].differentiate(y, false) - vec[1].differentiate(z, false);
            res[1] = vec[0].differentiate(z, false) - vec[2].differentiate(x, false);
            res[2] = vec[1].differentiate(x, false) - vec[0].differentiate(y, false);
        }
        if (origin && GLOBAL_FORMATTING.verbose) {
            if (GLOBAL_FORMATTING.output == algebra::detail::FormatSettings::Output::LATEX) {
                GLOBAL_FORMATTING << algebra::detail::LaTeX(std::string("\\nabla\\times").append(to_latex()).append("=").append(res.to_latex()));
            } else {
                GLOBAL_FORMATTING << "curl(" << *this << ") = " << res;
            }
        }
        return res;
    }

    Vector differentiate(const algebra::Variable& wrt, const bool origin = true) const {
        Vector res(size);

        if constexpr (requires(const T& obj, const algebra::Variable& variable, const bool origin) { obj.differentiate(variable, origin); }) {
            for (uint32_t i = 0; i < size; i++) {
                res[i] = vec[i].differentiate(wrt, false);
            }
        }
        if (origin && GLOBAL_FORMATTING.verbose) {
            algebra::detail::print_differentiate(*this, wrt, res);
        }
        return res;
    }

    Matrix<T> differentiate(const Vector<algebra::Variable>& wrt, const bool origin = true) const {
        Matrix<T> res(size, wrt.size);

        if constexpr (requires(const T& obj, const algebra::Variable& variable) { obj.differentiate(variable); }) {
            for (uint32_t i = 0; i < size; i++) {
                for (uint32_t j = 0; j < wrt.size; j++) {
                    res[i, j] = vec[i].differentiate(wrt[j], false);
                }
            }
            if (origin && GLOBAL_FORMATTING.verbose) {
                algebra::detail::print_differentiate(*this, wrt, res);
            }
        }
        return res;
    }

    std::string to_latex() const { return Matrix<T>(vec, transposed ? 1 : size, transposed ? size : 1, Matrix<T>::Type::VECTOR).to_latex(); }

    void serialize(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&serial_class), sizeof(serial_class));
        out.write(reinterpret_cast<const char*>(&transposed), sizeof(transposed));
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));

        for (const T& value : vec) {
            if constexpr (requires(const T& obj, std::ofstream& stream) { obj.serialize(stream); }) {
                value.serialize(out);
            } else {
                out.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
    }

    static Vector deserialize(std::ifstream& in) {
        detail::SerialClass type;
        in.read(reinterpret_cast<char*>(&type), sizeof(type));
        assert(type == serial_class);

        Vector res;
        in.read(reinterpret_cast<char*>(&res.transposed), sizeof(res.transposed));
        in.read(reinterpret_cast<char*>(&res.size), sizeof(res.size));
        res.vec.resize(res.size);

        for (T& value : res.vec) {
            if constexpr (requires(const T& obj, std::ifstream& stream) { obj.deserialize(stream); }) {
                value = T::deserialize(in);
            } else {
                in.read(reinterpret_cast<char*>(&value), sizeof(value));
            }
        }
        return res;
    }
};

namespace std {
    template <typename T>
    string to_string(const tensor::Vector<T>& vector) {
        return to_string(
            tensor::Matrix<T>(vector.vec, vector.transposed ? 1 : vector.size, vector.transposed ? vector.size : 1, tensor::Matrix<T>::Type::VECTOR));
    }
} // namespace std

template <typename T>
std::ostream& tensor::operator<<(std::ostream& out, const Vector<T>& vector) {
    return out << std::to_string(vector);
}

#pragma once
#include <set>
#include "algebra/algebra.hpp"

namespace tensor {
    inline algebra::detail::FormatSettings& GLOBAL_FORMATTING = algebra::GLOBAL_FORMATTING;

    template <typename>
    class Matrix;
    template <typename T>
    std::ostream& operator<<(std::ostream&, const Matrix<T>&);

    template <typename>
    class Vector;
    template <typename T>
    std::ostream& operator<<(std::ostream&, const Vector<T>&);

    std::map<algebra::Variable, algebra::Fraction> solve_linear_system(const std::vector<algebra::Equation>&, const std::string& = "gauss");

    namespace detail {
        enum class SerialClass : uint8_t { MATRIX, VECTOR };
    }
} // namespace tensor

#include "src/detail.hpp"
#include "src/matrix.hpp"
#include "src/vector.hpp"
#include "src/tensor.hpp"

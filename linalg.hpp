#pragma once
#include <set>
#include "algebra/algebra.hpp"

namespace linalg {
    inline algebra::detail::FormatSettings& GLOBAL_FORMATTING = algebra::GLOBAL_FORMATTING;

    template <typename>
    class Matrix;
    template <typename T>
    std::ostream& operator<<(std::ostream&, const Matrix<T>&);

    std::map<algebra::Variable, algebra::Fraction> solve_linear_system(const std::vector<algebra::Equation>&);
} // namespace linalg

#include "src/detail.hpp"
#include "src/matrix.hpp"
#include "src/linalg.hpp"

#pragma once

inline std::map<algebra::Variable, algebra::Fraction> linalg::solve_linear_system(const std::vector<algebra::Equation>& equations) {
    const auto [A, X, B] = Matrix<algebra::Variable>::from_equations(equations);
    const Matrix<algebra::Fraction> coefficients = A.augment(B);
    const std::vector<algebra::Fraction> values = coefficients.gauss_elimination();
    const int size = values.size();
    std::map<algebra::Variable, algebra::Fraction> res;

    for (int i = 0; i < size; i++) {
        res.emplace(X[i, 0], values[i]);
        GLOBAL_FORMATTING << X[i, 0] << '=' << values[i] << ' ';
    }
    GLOBAL_FORMATTING << std::endl;
    return res;
}

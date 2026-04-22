#pragma once

template <typename T = nullptr_t>
class tensor::Matrix {
    static constexpr auto serial_class = detail::SerialClass::MATRIX;

public:
    using value_type = T;
    enum class Type { AUGMENTED, DETERMINANT, NORMAL, VECTOR } type = Type::NORMAL;
    uint32_t type_param = -1, row, column;
    std::vector<std::vector<T>> matrix;

    Matrix() = default;

    Matrix(const uint32_t row, const uint32_t column, const T& value = T()) :
        row(row), column(column), matrix(std::vector(row, std::vector(column, value))) {}

    Matrix(std::initializer_list<T> list, const uint32_t row, const uint32_t column) : row(row), column(column), matrix(row, std::vector<T>(column)) {
        auto itr = list.begin();

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                assert(itr != list.end());
                matrix[i][j] = *itr;
                ++itr;
            }
        }
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list) :
        row(list.size()), column(list.begin()->size()), matrix(row, std::vector<T>(column)) {
        uint32_t i = 0;

        for (const std::initializer_list<T>& r : list) {
            assert(r.size() == column);
            uint32_t j = 0;

            for (const T& value : r) {
                matrix[i][j++] = value;
            }
            i++;
        }
    }

    Matrix(const std::vector<T>& vec, const uint32_t row, const uint32_t column, const Type type = Type::NORMAL) :
        type(type), row(row), column(column), matrix(row, std::vector<T>(column)) {
        const uint32_t size = vec.size();

        for (uint32_t i = 0, k = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                assert(k != size);
                matrix[i][j] = vec[k++];
            }
        }
    }

    Matrix(const std::vector<std::vector<T>>& matrix) : row(matrix.size()), column(matrix[0].size()), matrix(matrix) {}

    Matrix(const Matrix&) = default;

    template <typename U>
    Matrix(const Matrix<U>& mat) : Matrix(mat.row, mat.column) {
        *this = mat;
    }

    Matrix& operator=(const Matrix&) = default;

    template <typename U>
    Matrix& operator=(const Matrix<U>& mat) {
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                matrix[i][j] = static_cast<T>(mat[i, j]);
            }
        }
        return *this;
    }

    Matrix& operator=(Matrix&&) = default;

    template <typename U>
    Matrix& operator=(Matrix<U>&& mat) {
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                matrix[i][j] = static_cast<T&&>(mat[i, j]);
            }
        }
        return *this;
    }

    Matrix operator-() const {
        Matrix res(row, column);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[i, j] = -matrix[i][j];
            }
        }
        return res;
    }

    template <typename U>
    Matrix& operator+=(const U& value) {
        return *this = *this + value;
    }

    template <typename U, typename R = decltype(std::declval<T>() + std::declval<detail::unwrap_matrix_t<U>>())>
    Matrix<R> operator+(const U& value) const {
        Matrix<R> res(row, column);

        if constexpr (detail::is_matrix_v<U>) {
            assert(row == value.row && column == value.column);
        }
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                if constexpr (detail::is_matrix_v<U>) {
                    res[i, j] = matrix[i][j] + value[i, j];
                } else {
                    res[i, j] = matrix[i][j] + value;
                }
            }
        }
        return res;
    }

    template <typename U>
    Matrix& operator-=(const U& value) {
        return *this = *this - value;
    }

    template <typename U, typename R = decltype(std::declval<T>() - std::declval<detail::unwrap_matrix_t<U>>())>
    Matrix<R> operator-(const U& value) const {
        Matrix<R> res(row, column);

        if constexpr (detail::is_matrix_v<U>) {
            assert(row == value.row && column == value.column);
        }
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                if constexpr (detail::is_matrix_v<U>) {
                    res[i, j] = matrix[i][j] - value[i, j];
                } else {
                    res[i, j] = matrix[i][j] - value;
                }
            }
        }
        return res;
    }

    template <typename U>
    Matrix& operator*=(const U& value) {
        return *this = *this * value;
    }

    template <typename U, typename t = decltype(std::declval<T>() * std::declval<detail::unwrap_matrix_t<U>>()),
              typename R = decltype(std::declval<t>() + std::declval<t>())>
    Matrix<R> operator*(const U& value) const {
        if constexpr (detail::is_matrix_v<U>) {
            assert(column == value.row);
            Matrix<R> res(row, value.column);

            for (uint32_t i = 0; i < row; i++) {
                for (uint32_t j = 0; j < value.column; j++) {
                    for (uint32_t k = 0; k < column; k++) {
                        res[i, j] += matrix[i][k] * value[k, j];
                    }
                }
            }
            return res;
        } else {
            Matrix<R> res(row, column);

            for (uint32_t i = 0; i < row; i++) {
                for (uint32_t j = 0; j < column; j++) {
                    res[i, j] = matrix[i][j] * value;
                }
            }
            return res;
        }
    }

    template <typename U, typename t = decltype(std::declval<T>() * std::declval<U>()), typename R = decltype(std::declval<t>() + std::declval<t>())>
    Vector<R> operator*(const Vector<U>& value) const {
        assert(column == value.size);
        Vector<R> res(row);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[i] += matrix[i][j] * value[j];
            }
        }
        return res;
    }

    template <typename U>
        requires(!detail::is_matrix_v<U>)
    Matrix& operator/=(const U& value) {
        return *this = *this / value;
    }

    template <typename U, typename R = decltype(std::declval<T>() / std::declval<detail::unwrap_matrix_t<U>>())>
        requires(!detail::is_matrix_v<U>)
    Matrix<R> operator/(const U& value) const {
        Matrix<R> res(row, column);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[i, j] = matrix[i][j] / value;
            }
        }
        return res;
    }

    bool operator==(const Matrix& other) const { return matrix == other.matrix; }

    std::vector<T>& operator[](const uint32_t i) { return matrix[i]; }

    const std::vector<T>& operator[](const uint32_t i) const { return matrix[i]; }

    T& operator[](const uint32_t i, const uint32_t j) { return matrix[i][j]; }

    const T& operator[](const uint32_t i, const uint32_t j) const { return matrix[i][j]; }

    static Matrix make_identity(const uint32_t row) {
        Matrix res(row, row);

        for (uint32_t i = 0; i < row; i++) {
            res[i, i] = 1;
        }
        return res;
    }

    static std::tuple<Matrix<algebra::Fraction>, Matrix<algebra::Variable>, Matrix<algebra::Fraction>>
    from_equations(const std::vector<algebra::Equation>& equations) {
        const uint32_t size = equations.size();
        Matrix<algebra::Fraction> B(size, 1);
        std::set<algebra::Variable> variables;

        for (uint32_t i = 0; i < size; i++) {
            for (const algebra::Variable& variable : equations[i].lhs.terms) {
                variables.insert(variable.basis());
            }
            B[i, 0] = static_cast<algebra::Fraction>(equations[i].rhs);
        }
        Matrix<algebra::Fraction> A(size, variables.size());

        for (uint32_t i = 0; i < size; i++) {
            for (const algebra::Variable& variable : equations[i].lhs.terms) {
                A[i, std::distance(variables.begin(), variables.find(variable.basis()))] = variable.coefficient;
            }
        }
        return {A, Matrix<algebra::Variable>(std::vector(variables.begin(), variables.end()), variables.size(), 1), B};
    }

    template <typename U>
    static Matrix<U> differentiate(const U& scalar, const Matrix<algebra::Variable>& wrt, const bool origin = true) {
        Matrix<U> res(wrt.row, wrt.column);

        if constexpr (requires(const T& obj, const algebra::Variable& variable) { obj.differentiate(variable); }) {
            for (uint32_t i = 0; i < wrt.row; i++) {
                for (uint32_t j = 0; j < wrt.column; j++) {
                    res[i, j] = scalar.differentiate(wrt[i, j], false);
                }
            }
        }
        if (origin && GLOBAL_FORMATTING.verbose) {
            algebra::detail::print_differentiate(scalar, wrt, res);
        }
        return res;
    }

    Matrix differentiate(const algebra::Variable& wrt, const bool origin = true) const {
        Matrix res(row, column);

        if constexpr (requires(const T& obj, const algebra::Variable& variable, const bool origin) { obj.differentiate(variable, origin); }) {
            for (uint32_t i = 0; i < row; i++) {
                for (uint32_t j = 0; j < column; j++) {
                    res[i, j] = matrix[i][j].differentiate(wrt, false);
                }
            }
        }
        if (origin && GLOBAL_FORMATTING.verbose) {
            algebra::detail::print_differentiate(*this, wrt, res);
        }
        return res;
    }

    Matrix transpose() const {
        Matrix res(column, row);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[j, i] = matrix[i][j];
            }
        }
        return res;
    }

    Matrix augment(const Matrix& mat) const {
        assert(row == mat.row);
        Matrix res(row, column + mat.column);

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                res[i, j] = matrix[i][j];
            }
            for (uint32_t j = 0; j < mat.column; j++) {
                res[i, column + j] = mat[i, j];
            }
        }
        res.type = Type::AUGMENTED;
        res.type_param = mat.column;
        return res;
    }

    T determinant() const {
        assert(row == column);
        GLOBAL_FORMATTING << "Determinant:" << std::endl;
        Matrix temp = *this;
        temp.type = Type::DETERMINANT;
        Matrix mat = temp.echelon_form();
        T res = 1;

        for (uint32_t i = 0; i < row; i++) {
            res *= mat[i, i];
        }
        return mat.type_param % 2 == 0 ? res : -res;
    }


    Matrix echelon_form() const {
        Matrix res = *this;
        uint32_t pivot = 0;
        GLOBAL_FORMATTING << "Echelon Form:" << std::endl << res;

        if (res.type == Type::DETERMINANT) {
            res.type_param = 0;
        }
        for (uint32_t cnt = 0; cnt < column && pivot < row; cnt++) {
            uint32_t pivot_row = pivot;

            while (pivot_row < row && res[pivot_row, cnt] == 0) {
                pivot_row++;
            }
            if (pivot_row == row) {
                continue;
            }
            if (pivot_row != pivot) {
                for (uint32_t j = 0; j < column; j++) {
                    std::swap(res[pivot_row, j], res[pivot, j]);
                }
                if (res.type == Type::DETERMINANT) {
                    ++res.type_param;
                }
            }
            for (uint32_t i = pivot + 1; i < row; i++) {
                T factor = res[i, cnt] / res[pivot, cnt];

                for (uint32_t j = cnt; j < column; j++) {
                    res[i, j] -= factor * res[pivot, j];
                }
            }
            pivot++;
            GLOBAL_FORMATTING << res;
        }
        return res;
    }

    std::vector<T> gauss_elimination() const {
        GLOBAL_FORMATTING << "Gauss Elimination:" << std::endl;
        Matrix res = echelon_form();
        std::vector<T> solution(row);

        for (uint32_t i = 0; i < row; i++) {
            bool flag = true;

            for (uint32_t j = 0; j < column - 1; j++) {
                if (res[i, j] != 0) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                if (res[i, column - 1] == 0) {
                    GLOBAL_FORMATTING << "Infinitely many solutions" << std::endl;
                } else {
                    GLOBAL_FORMATTING << "No solution" << std::endl;
                }
                return {};
            }
        }
        for (int i = row - 1; i >= 0; i--) {
            T sum = res[i, column - 1];

            for (uint32_t j = i + 1; j < row; j++) {
                sum -= res[i, j] * solution[j];
            }
            solution[i] = sum / res[i, i];
        }
        return solution;
    }

    uint32_t rank() const {
        Matrix res = echelon_form();
        return std::ranges::count(res.matrix, true, [](const std::vector<T>& row) -> bool {
            return !std::ranges::all_of(row, [](const T& value) -> bool { return value == T(); });
        });
    }

    std::vector<T> cramer_rule(const Matrix& rhs) const {
        T det = determinant();
        std::vector<T> solution;
        solution.reserve(column);

        for (uint32_t i = 0; i < column; i++) {
            Matrix res = *this;

            for (uint32_t j = 0; j < row; j++) {
                res[j, i] = rhs[j, 0];
            }
            solution.push_back(res.determinant() / det);
        }
        return solution;
    }

    Matrix inverse() const {
        assert(determinant() != 0);
        GLOBAL_FORMATTING << "Inverse:" << std::endl;
        GLOBAL_FORMATTING << *this;
        Matrix res(row, column), aug_matrix = augment(make_identity(row));
        GLOBAL_FORMATTING << aug_matrix;
        aug_matrix = aug_matrix.echelon_form();

        for (int i = row - 1; i >= 0; i--) {
            T pivot = aug_matrix[i, i];
            assert(pivot != 0);

            for (uint32_t j = 0; j < row * 2; j++) {
                aug_matrix[i, j] /= pivot;
            }
            for (uint32_t k = 0; k < i; k++) {
                T factor = aug_matrix[k, i];

                for (uint32_t j = 0; j < row * 2; j++) {
                    aug_matrix[k, j] -= factor * aug_matrix[i, j];
                }
            }
            GLOBAL_FORMATTING << aug_matrix;
        }
        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < row; j++) {
                res[i][j] = aug_matrix[i, j + row];
            }
        }
        return res;
    }

    std::string to_latex() const {
        std::string res;

        switch (type) {
        case Type::AUGMENTED:
            res.append("\\left(\n\\begin{array}{").append(column - type_param, 'c').append("|").append(type_param, 'c').append("}\n");
            break;

        case Type::DETERMINANT:
            res.append("\\begin{vmatrix}\n");
            break;

        case Type::VECTOR:
        case Type::NORMAL:
            res.append("\\begin{pmatrix}\n");
            break;
        }

        for (uint32_t i = 0; i < row; i++) {
            for (uint32_t j = 0; j < column; j++) {
                if constexpr (requires(const T& obj) { obj.to_latex(); }) {
                    res.append(matrix[i][j].to_latex());
                } else {
                    res.append(std::to_string(matrix[i][j]));
                }
                if (j != column - 1) {
                    res.append(" & ");
                } else {
                    res.append("\\\\\n");
                }
            }
        }
        switch (type) {
        case Type::AUGMENTED:
            res.append("\\end{array}\n\\right)");
            break;

        case Type::DETERMINANT:
            res.append("\\end{vmatrix}");
            break;

        case Type::VECTOR:
        case Type::NORMAL:
            res.append("\\end{pmatrix}");
            break;
        }
        if (type != Type::VECTOR) {
            return res.append("_{").append(std::to_string(row)).append("\\times ").append(std::to_string(column)).append("}\n");
        }
        return res;
    }

    std::string to_html() const {
        std::string res;

        switch (type) {
        case Type::DETERMINANT:
            res.append("<msub><mrow><mo fence='true' stretchy='true'>|</mo><mtable>");
            break;

        case Type::NORMAL:
        case Type::AUGMENTED:
            res.append("<msub><mrow>");

        case Type::VECTOR:
            res.append("<mo fence='true' stretchy='true'>(</mo><mtable>");
            break;
        }

        for (uint32_t i = 0; i < row; i++) {
            res.append("<mtr>");

            for (uint32_t j = 0; j < column; j++) {
                if (type == Type::AUGMENTED && j == column - type_param - 1) {
                    res.append("<mtd style='border-right: 1px solid black; padding-right: 10px;'>");
                } else {
                    res.append("<mtd>");
                }
                if constexpr (requires(const T& obj) { obj.to_html(); }) {
                    res.append(matrix[i][j].to_html());
                } else {
                    res.append(std::to_string(matrix[i][j]));
                }
                res.append("</mtd>");
            }
            res.append("</mtr>");
        }
        switch (type) {
        case Type::DETERMINANT:
            res.append("</mtable><mo fence='true' stretchy='true'>|</mo>");
            break;

        case Type::AUGMENTED:
        case Type::VECTOR:
        case Type::NORMAL:
            res.append("</mtable><mo fence='true' stretchy='true'>)</mo>");
            break;
        }
        if (type != Type::VECTOR) {
            return res.append("</mrow><mrow><mn>")
                .append(std::to_string(row))
                .append("</mn><mo>&times;</mo><mn>")
                .append(std::to_string(column))
                .append("</mn></mrow></msub>");
        }
        return res;
    }

    void serialize(std::ofstream& out) const {
        out.write(reinterpret_cast<const char*>(&serial_class), sizeof(serial_class));
        out.write(reinterpret_cast<const char*>(&type), sizeof(type));
        out.write(reinterpret_cast<const char*>(&type_param), sizeof(type_param));
        out.write(reinterpret_cast<const char*>(&row), sizeof(row));
        out.write(reinterpret_cast<const char*>(&column), sizeof(column));

        for (const std::vector<T>& element : matrix) {
            for (const T& value : element) {
                if constexpr (requires(const T& obj, std::ofstream& stream) { obj.serialize(stream); }) {
                    value.serialize(out);
                } else {
                    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
                }
            }
        }
    }

    static Matrix deserialize(std::ifstream& in) {
        detail::SerialClass type;
        in.read(reinterpret_cast<char*>(&type), sizeof(type));
        assert(type == serial_class);

        Matrix res;
        in.read(reinterpret_cast<char*>(&res.type), sizeof(res.type));
        in.read(reinterpret_cast<char*>(&res.type_param), sizeof(res.type_param));
        in.read(reinterpret_cast<char*>(&res.row), sizeof(res.row));
        in.read(reinterpret_cast<char*>(&res.column), sizeof(res.column));
        res.matrix.resize(res.row, std::vector<T>(res.column));

        for (std::vector<T>& element : res.matrix) {
            for (T& value : element) {
                if constexpr (requires(const T& obj, std::ifstream& stream) { obj.deserialize(stream); }) {
                    value = T::deserialize(in);
                } else {
                    in.read(reinterpret_cast<char*>(&value), sizeof(value));
                }
            }
        }
        return res;
    }
};

template <typename T, typename U, typename R = decltype(std::declval<U>() + std::declval<T>())>
    requires(!tensor::detail::is_matrix_v<T>)
tensor::Matrix<R> operator+(const T& value, const tensor::Matrix<U>& matrix) {
    return matrix + value;
}

template <typename T, typename U, typename R = decltype(-std::declval<U>() + std::declval<T>())>
    requires(!tensor::detail::is_matrix_v<T>)
tensor::Matrix<R> operator-(const T& value, const tensor::Matrix<U>& matrix) {
    return -matrix + value;
}

template <typename T, typename U, typename R = decltype(std::declval<U>() * std::declval<T>())>
    requires(!tensor::detail::is_matrix_v<T>)
tensor::Matrix<R> operator*(const T& value, const tensor::Matrix<U>& matrix) {
    return matrix * value;
}

template <typename T, typename U, typename R = decltype(std::declval<T>() / std::declval<U>())>
    requires(!tensor::detail::is_matrix_v<T>)
tensor::Matrix<R> operator/(const T& value, const tensor::Matrix<U>& matrix) {
    tensor::Matrix<R> res(matrix.row, matrix.column);

    for (uint32_t i = 0; i < matrix.row; i++) {
        for (uint32_t j = 0; j < matrix.column; j++) {
            res[i, j] = value / matrix[i, j];
        }
    }
    return res;
}

namespace std {
    template <typename T>
    std::string to_string(const tensor::Matrix<T>& matrix) {
        uint32_t padding = 0;
        tensor::Matrix<std::string> format(matrix.row, matrix.column);

        for (uint32_t i = 0; i < matrix.row; i++) {
            for (uint32_t j = 0; j < matrix.column; j++) {
                format[i, j] = std::to_string(matrix[i, j]);
                padding = std::max(padding, static_cast<uint32_t>(format[i, j].size()));
            }
        }
        padding += 2;
        const uint32_t total_width = matrix.column * padding + (matrix.column - 1);
        std::string middle(total_width, ' ');

        if (matrix.type == tensor::Matrix<T>::Type::AUGMENTED) {
            const uint32_t separation = matrix.column - matrix.type_param;
            middle[separation * padding + (separation - 1)] = '|';
        }
        const uint32_t left = padding / 2, right = padding - left;
        std::string res("\n"), border("+"), empty_space("|");
        border.append(left, '-').append(total_width - left - right, ' ').append(right, '-').push_back('+');
        empty_space.append(middle).push_back('|');

        if (matrix.type != tensor::Matrix<T>::Type::DETERMINANT) {
            res.append(border);
        }
        for (uint32_t i = 0; i < format.row; i++) {
            res.append("\n|");

            for (uint32_t j = 0; j < format.column; j++) {
                const std::string& val = format[i, j];
                const uint32_t remaining = padding - val.size();
                res.append(std::string(remaining / 2, ' ')).append(val).append(std::string(remaining - remaining / 2, ' '));

                if (j < format.column - 1) {
                    if (matrix.type == tensor::Matrix<T>::Type::AUGMENTED && j == matrix.column - matrix.type_param - 1) {
                        res.push_back('|');
                    } else {
                        res.push_back(' ');
                    }
                }
            }
            res.append("|");

            if (i < format.row - 1) {
                res.append("\n").append(empty_space);
            }
        }

        if (matrix.type != tensor::Matrix<T>::Type::DETERMINANT) {
            res.append("\n").append(border);
        }
        if (matrix.type != tensor::Matrix<T>::Type::VECTOR) {
            res.append(" ").append(std::to_string(matrix.row)).append("x").append(std::to_string(matrix.column));
        }
        return res.append("\n");
    }
} // namespace std

template <typename T>
std::ostream& tensor::operator<<(std::ostream& out, const Matrix<T>& matrix) {
    return out << std::to_string(matrix);
}

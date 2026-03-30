#pragma once

namespace tensor::detail {
    template <typename>
    struct is_matrix : std::false_type {};

    template <typename T>
    struct is_matrix<Matrix<T>> : std::true_type {};

    template <typename T>
    inline constexpr bool is_matrix_v = is_matrix<T>::value;

    template <typename U>
    struct unwrap_matrix {
        using type = U;
    };

    template <typename T>
    struct unwrap_matrix<Matrix<T>> {
        using type = T;
    };

    template <typename U>
    using unwrap_matrix_t = unwrap_matrix<U>::type;
} // namespace tensor::detail

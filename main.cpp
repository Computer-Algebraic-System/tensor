#include "tensor.hpp"

using namespace algebra;
using namespace tensor;

algebra::detail::FormatSettings& out = tensor::GLOBAL_FORMATTING;

int main() {
    // tensor::GLOBAL_FORMATTING.toggle_file("output.txt");
    // tensor::GLOBAL_FORMATTING.toggle_latex("latex.tex");
    Variable x("x"), y("y"), z("z"), x1("x1"), x2("x2"), x3("x3"), x4("x4"), i1("i1"), i2("i2"), i3("i3");

    out << Matrix<Fraction>{{0.5, 2, 7}, {3, -1, 9}}.transpose() << std::endl;
    out << 2 * Matrix<Fraction>{{1, -5}, {3, 7}} << std::endl;
    out << Matrix<Fraction>{{2, 3}, {-1, 7}} + Matrix<Fraction>{{7, -8}, {2, 0}} << std::endl;
    out << Matrix<Fraction>{{1, 2}, {3, 4}, {5, 6}} * Matrix<Fraction>{{2, 5}, {6, 8}} << std::endl;
    out << Matrix<Fraction>{{1, 1}, {2, 2}} * Matrix<Fraction>{{-1, 1}, {1, -1}} << std::endl;

    solve_linear_system({
        -x1 + x2 + 2 * x3 == 2,
        3 * x1 - x2 + x3 == 6,
        -x1 + 3 * x2 + 4 * x3 == 4,
    });
    solve_linear_system({
        3 * x1 + 2 * x2 + 2 * x3 - 5 * x4 == 8,
        0.6 * x1 + 1.5 * x2 + 1.5 * x3 - 5.4 * x4 == 2.7,
        1.2 * x1 - 0.3 * x2 - 0.3 * x3 + 2.4 * x4 == 2.1,
    });
    solve_linear_system({
        3 * x1 + 2 * x2 + x3 == 3,
        2 * x1 + x2 + x3 == 0,
        6 * x1 + 2 * x2 + 4 * x3 == 6,
    });
    solve_linear_system({
        i1 - i2 + i3 == 0,
        2 * i2 + i3 == 8,
        2 * i2 + 5 * i3 == 18,
    });
    out << Matrix<Fraction>({{3, 0, 2, 2}, {-6, 42, 24, 54}, {21, -21, 0, -15}}).rank();
    solve_linear_system(
        {
            3 * x + 7 * y + 8 * z == -13,
            2 * x + 9 * z == -5,
            -4 * x + y - 26 * z == 2,
        },
        "cramer");
    out << Matrix<Fraction>({{-1, 1, 2}, {3, -1, 1}, {-1, 3, 4}}).inverse() << std::endl;
    out << Matrix<Fraction>({{2, 0, -1}, {5, 1, 0}, {0, 1, 3}}).inverse() << std::endl;
    return 0;
}

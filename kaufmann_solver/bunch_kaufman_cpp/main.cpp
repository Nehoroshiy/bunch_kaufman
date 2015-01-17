#include "linear_solvers.h"
#include "tests.h"

int main(int argc, char **argv) {
    auto N = 10;
    if (argc == 2)
        N = atoi(argv[1]);
    //auto N = 12;
    Vector original_result = Vector::I(N);
    /*for (int i = 0; i < N; i++)
        original_result[i] = i + 1;*/
    std::cout << "orig: [";
    for (int i = 0; i < N; i++) {
        std::cout << original_result[i] << " ";
    }
    std::cout << "]" << std::endl;
    
    Matrix hilbert_wrapper = hilbert_matrix_m(N);
    Vector free_variables = hilbert_wrapper * original_result;
    std::cout << "free_variables: [";
    for (int i = 0; i < N; i++) {
        std::cout << free_variables[i] << " ";
    }
    std::cout << "]" << std::endl;
    bunch_kaufman_solve(hilbert_wrapper.matrix, free_variables.data, N);

    std::cout << "solution: [";
    for (int i = 0; i < N; i++) {
        std::cout << free_variables[i] << " ";
    }
    std::cout << "]" << std::endl;
    auto delta = (original_result - free_variables);
    std::cout << "delta: [";
    for (int i = 0; i < N; i++) {
        std::cout << delta[i] << " ";
    }
    std::cout << "]" << std::endl;
    auto relative_error = (original_result - free_variables).euclid_norm() / original_result.euclid_norm();
    std::cout << (original_result - free_variables).euclid_norm() << "::" << original_result.euclid_norm() << std::endl;
    std::cout << "Relative_error: " << relative_error << std::endl;
    //auto min_size = 5;
    //auto max_size = 300;
    //bunch_kaufmann_full_test(min_size, max_size, "bunch_kaufman_full_test.txt");
    return 0;
}
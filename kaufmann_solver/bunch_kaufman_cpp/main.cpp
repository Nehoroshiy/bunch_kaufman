#include "bunch_kaufman.h"
#include "tests.h"

int main() {
    auto SIZE = 102;
    auto m = hilbert_matrix(SIZE);
    auto PL = Matrix::I(SIZE);
    bunch_kaufman(m, PL);
    print_matrix(PL, "PL matrix");
    auto permutation = distinct_permutation_and_lower_triangular(PL);

    //std::cout << m[2][2] << std::endl;
    print_matrix(m, "tridiagonal_matrix");
    print_matrix(PL, "L matrix");
    print_vector<int>(permutation);

    auto PL_T = PL.transpose();
    auto permutation_T = permutation_transpose(permutation);
    auto check_matrix = PL * m * PL_T;
    check_matrix.permute_rows(permutation);
    check_matrix.permute_columns(permutation_T);
    print_matrix(check_matrix, "check matrix");
    auto hilb = hilbert_matrix(SIZE);
    std::cout << "Frobenius norm: " << (check_matrix - hilb).frobenius_norm() << std::endl; 
    return 0;
}
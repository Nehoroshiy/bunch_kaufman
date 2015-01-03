#include "bunch_kaufman.h"
#include "tests.h"

int main() {
    /*auto SIZE = 20;
    double *m = hilbert_matrix(SIZE);
    double *PL = identity_matrix(SIZE);
    bunch_kaufman(m, PL, SIZE);
    print_matrix(PL, SIZE, "PL matrix");
    auto permutation = distinct_permutation_and_lower_triangular(PL, SIZE);
    print_matrix(m, SIZE, "tridiagonal_matrix");
    print_matrix(PL, SIZE, "L matrix");
    print_vector<int>(permutation);
    auto permutation_T = permutation_transpose(permutation);
    double *check_matrix = matrix_conjugation(m, PL, SIZE);
    Matrix check_matrix_wrapper = Matrix(check_matrix, SIZE);
    check_matrix_wrapper.permute_rows(permutation);
    check_matrix_wrapper.permute_columns(permutation_T);
    print_matrix(check_matrix, SIZE, "check_matrix");
    Matrix hilb = Matrix(hilbert_matrix(SIZE), SIZE);
    std::cout << "Frobenius norm: " << (check_matrix_wrapper - hilb).frobenius_norm() << std::endl;
    check_matrix_wrapper.reject();*/

    auto SIZE = 300;
    double *m = hilbert_matrix(SIZE);
    double *PL = identity_matrix(SIZE);
    
    bunch_kaufman(m, PL, SIZE);
    double *check_matrix = matrix_conjugation(m, PL, SIZE);
    Matrix check_matrix_wrapper = Matrix(check_matrix, SIZE);
    //print_matrix(check_matrix, SIZE, "check_matrix");
    Matrix hilb = Matrix(hilbert_matrix(SIZE), SIZE);
    std::cout << "Frobenius norm: " << (check_matrix_wrapper - hilb).frobenius_norm() << std::endl;
    //std::cout << "Frobenius norm: " << (check_matrix_wrapper - hilb).frobenius_norm_exact() << std::endl;
    std::cout << "Row norm: " << (check_matrix_wrapper - hilb).norm() << std::endl;
    check_matrix_wrapper.reject();

    
    /*
    auto SIZE = 10;
    double *A = new double[SIZE * SIZE];
    double *B = new double[SIZE * SIZE];
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            A[i * SIZE + j] = B[i * SIZE + j] = 1.0;
    //double *A = identity_matrix(SIZE);
    //double *B = identity_matrix(SIZE);
    double *C = matrix_conjugation(A, B, SIZE);
    print_matrix(C, SIZE, "NNN");
    */
    

    /*auto SIZE = 10;
    auto m = hilbert_matrix(SIZE);
    auto M = Matrix(m, SIZE);
    auto PL = Matrix::I(SIZE);
    bunch_kaufman(m, PL.matrix, SIZE);
    print_matrix(PL.matrix, SIZE, "PL matrix");
    auto permutation = distinct_permutation_and_lower_triangular(PL);

    //std::cout << m[2][2] << std::endl;
    print_matrix(m, SIZE, "tridiagonal_matrix");
    print_matrix(PL.matrix, SIZE, "L matrix");
    print_vector<int>(permutation);

    auto PL_T = PL.transpose();
    auto permutation_T = permutation_transpose(permutation);
    auto check_matrix = PL * M * PL_T;
    check_matrix.permute_rows(permutation);
    check_matrix.permute_columns(permutation_T);
    print_matrix_m(check_matrix, "check matrix");
    auto hilb = Matrix(hilbert_matrix(SIZE), SIZE);
    std::cout << "Frobenius norm: " << (check_matrix - hilb).frobenius_norm() << std::endl;*/ 
    return 0;
}
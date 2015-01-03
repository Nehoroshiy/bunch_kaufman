#include "bunch_kaufman.h"

double normalize(double val) {
    if (fabs(val) < TOLERANCE) return 0.0;
    return val;
}

std::vector<int> permute_ident(size_t N) {
    auto permutation = std::vector<int>(N);
    for (size_t i = 0; i < N; i++)
        permutation[i] = i;
    return permutation;
}

std::vector<int> permutation_transpose(std::vector<int> &permutation) {
    auto transpose = std::vector<int>(permutation.size());
    for (size_t i = 0; i < permutation.size(); i++)
        transpose[permutation[i]] = i;
    return transpose;
}

// ????????????????????????????
std::vector<int> compose_permutations(std::vector<int> &outer, std::vector<int> &inner) {
    auto permutation = std::vector<int>(inner.size());
    for (size_t i = 0; i < inner.size(); i++) {
        permutation[i] = outer[inner[i]];
    }
    return permutation;
}

Matrix inverse_1_2(Matrix &small_matrix) {
    if (small_matrix.dim() == 1) {
        auto inverse = Matrix::I(1);
        inverse[0][0] /= small_matrix[0][0];
        return inverse;
    }
    auto a = small_matrix[0][0];
    auto b = small_matrix[0][1];
    auto c = small_matrix[1][0];
    auto d = small_matrix[1][1];
    auto det = a * d - b * c;
    if (det == 0) exit(0);
    auto inverse = Matrix::zeros(2);
    inverse[0][0] = d / det;
    inverse[0][1] = -c / det;
    inverse[1][0] = -b / det;
    inverse[1][1] = a / det;
    return inverse;
}

void bunch_kaufman(double *input_matrix, double *pl_factor, size_t N, double alpha) {
    Matrix matrix = Matrix(input_matrix, N);
    Matrix PL = Matrix(pl_factor, N);
    //print_matrix(matrix, "initial matrix");
    auto sum = 0;
    while (sum < N) {
        //std::cout << "sum: " << sum << std::endl;
        auto max_with_idx_diag = matrix.max_in_diagonal(sum);
        auto transposition = std::pair<int, int>(sum, max_with_idx_diag.second);
        matrix.exchange_rows(transposition.first, transposition.second);
        matrix.exchange_columns(transposition.second, transposition.first);

        PL.exchange_columns(transposition.first, transposition.second);

        auto max_from_column = matrix.max_in_column(sum, sum);
        auto lambda_val = max_from_column.first;
        auto idx = max_from_column.second;
        auto j_idx = 0;
        int n_k = 0;
        auto permutation = permute_ident(N);
        bool perm_active = false;
        if (fabs(matrix[sum][sum]) >= alpha * lambda_val) {
            n_k = 1;
            if (N <= sum + n_k) break;
        } else {
            auto max_from_column_2 = matrix.max_in_column(idx, sum, 0, true);
            auto sigma_val = max_from_column_2.first;
            j_idx = max_from_column_2.second;
            if (sigma_val * fabs(matrix[sum][sum]) >= alpha * lambda_val * lambda_val) {
                n_k = 1;
                if (N <= sum + n_k) break;
            } else {
                if (fabs(matrix[idx][idx]) >= alpha * sigma_val) {
                    n_k = 1;
                    if (N <= sum + n_k) break;
                    permutation[sum] = idx;
                    permutation[idx] = sum;
                    perm_active = true;
                } else {
                    n_k = 2;
                    if (N <= sum + n_k) break;
                    auto tempr = permutation[sum + 1];
                    permutation[sum + 1] = permutation[j_idx];
                    permutation[j_idx] = tempr;
                    tempr = permutation[sum + 2];
                    permutation[sum + 2] = permutation[idx];
                    permutation[idx] = tempr;
                    perm_active = true;
                }
            }
        }
        //print_matrix(matrix.matrix, N, "before exchanges");
        if (n_k == 1 && perm_active) {
            matrix.exchange_rows(sum, idx);
            matrix.exchange_columns(sum, idx);
            PL.exchange_columns(sum, idx);
        } else if (n_k == 2) {
            //std::cout << "sum + 1: " << sum + 1 << ", j_idx: " << j_idx << std::endl;
            //std::cout << "sum + 1: " << sum + 1 << ", idx: " << idx << std::endl;
            matrix.exchange_rows(sum, j_idx);
            matrix.exchange_rows(sum + 1, idx);
            matrix.exchange_columns(sum, j_idx);
            matrix.exchange_columns(sum + 1, idx);
            PL.exchange_columns(sum, j_idx);
            PL.exchange_columns(sum + 1, idx);
        }
        //print_matrix(matrix.matrix, N, "after exchanges");
        auto T_k = Matrix::zeros(n_k);
        for (int i = 0; i < n_k; i++)
            for (int j = 0; j < n_k; j++)
                T_k[i][j] = matrix[sum + i][sum + j];
        auto T_k_inverse = inverse_1_2(T_k);
        if (n_k == 1) {
            // simple case
            // form vector B_k
            auto b_height = N - sum - 1;
            auto B_k = Vector::zeros(N);
            for (size_t i = sum + 1; i < N; i++)
                B_k[i] = matrix[i][sum];

            B_k.scale(-T_k_inverse[0][0]);
            auto B_k_inverse = Vector(B_k);
            B_k_inverse.scale(-1.0);
            // multiply by left
            double backed_row[N];
            long double backed_long[N];
            for (size_t i = 0; i < N; i++)
                backed_long[i] = matrix[sum][i];
            memcpy(backed_row, matrix[sum], sizeof(double) * N);
            for (size_t i = sum + 1; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    if (j == sum) {
                        matrix[i][j] = 0.0;
                        continue;
                    }
                    long double precision_saver = static_cast<long double>(matrix[i][j]);
                    precision_saver += static_cast<long double>(B_k[i]) * backed_long[j];
                    matrix[i][j] = static_cast<double>(precision_saver);
                    #ifdef NORMALIZE_WITH_TOLERANCE
                    normalize(matrix[i][j]);
                    #endif
                }
            }
            // multiply by right
            for (size_t i = 0; i < N; i++) {
                backed_row[i] = matrix[i][sum];
                backed_long[i] = static_cast<long double>(matrix[i][sum]);
            }
            for (size_t j = sum + 1; j < N; j++)
                matrix[sum][j] = 0;
            for (size_t i = sum + 1; i < N; i++) {
                for (size_t j = sum + 1; j < N; j++) {
                    long double precision_saver = static_cast<long double>(matrix[i][j]);
                    precision_saver += backed_long[i] * static_cast<long double>(B_k[j]);
                    matrix[i][j] = static_cast<double>(precision_saver);
                    //matrix[i][j] += backed_row[i] * B_k[j];
                    #ifdef NORMALIZE_WITH_TOLERANCE
                    matrix[i][j] = normalize(matrix[i][j]);
                    #endif
                }
            }
            /*for (size_t i = 0; i < N; i++)
                backed_row[i] = 0;
            for (size_t i = 0; i < N; i++) {
                for (size_t j = sum + 1; j < N; j++) {
                    backed_row[i] += matrix[i][j] * B_k_inverse[j];
                }
            }
            for (size_t i = 0; i < N; i++)
                matrix[i][sum] += backed_row[i];*/
            // and PL
            
            //PL.permute_columns(permutation_T);
            // multiply by right
            for (size_t i = 0; i < N; i++)
                backed_row[i] = 0.0;
            for (size_t i = 0; i < N; i++) {
                for (size_t j = sum + 1; j < N; j++) {
                    backed_row[i] += PL[i][j] * B_k_inverse[j];
                }
            }
            for (size_t i = 0; i < N; i++)
                PL[i][sum] += backed_row[i];
            // multiply by right
            /*for (size_t i = 0; i < N; i++) {
                backed_row[i] = PL[i][sum];
                backed_long[i] = static_cast<long double>(PL[i][sum]);
            }
            for (size_t i = 0; i < N; i++) {
                for (size_t j = sum + 1; j < N; j++) {
                    long double precision_saver = static_cast<long double>(PL[i][j]);
                    precision_saver += backed_long[i] * static_cast<long double>(B_k_inverse[j]);
                    PL[i][j] = static_cast<double>(precision_saver);
                    //matrix[i][j] += backed_row[i] * B_k[j];
                    #ifdef NORMALIZE_WITH_TOLERANCE
                    PL[i][j] = normalize(PL[i][j]);
                    #endif
                }
            }*/
            /*for (size_t i = 0; i < N; i++) {
                for (size_t j = sum + 1; j < N; j++) {
                    PL[i][j] += backed_row[i] * B_k_inverse[j];
                    #ifdef NORMALIZE_WITH_TOLERANCE
                    PL[i][j] = normalize(PL[i][j]);
                    #endif
                }
            }*/
            sum += 1;
            /*for (int ttt = 0; ttt < 3; ttt++) print_straight_line(50);
            std::cout << "sum: " << sum << std::endl;
            print_matrix(input_matrix, N, "tridiagonal_matrix");
            print_matrix(pl_factor, N, "PL matrix");
            double *check_matrix = matrix_conjugation(input_matrix, pl_factor, N);
            Matrix check_matrix_wrapper = Matrix(check_matrix, N);
            print_matrix(check_matrix, N, "check_matrix");
            Matrix hilb = Matrix(hilbert_matrix(N), N);
            print_matrix(hilb.matrix, N, "hilbert_matrix");
            Matrix delta = check_matrix_wrapper - hilb;
            print_matrix(delta.matrix, N, "delta_matrix");
            std::cout << "Frobenius norm: " << (check_matrix_wrapper - hilb).frobenius_norm() << std::endl;
            std::cout << "Row norm: " << (check_matrix_wrapper - hilb).norm() << std::endl;
            check_matrix_wrapper.reject();
            for (int ttt = 0; ttt < 3; ttt++) print_straight_line(50);*/
            //print_matrix(matrix, "tridiagonal matrix");
            //print_matrix(PL, "PL");
        } else {
            /*// simple case
            // form vector B_k
            auto b_height = N - sum - 1;
            auto B_k = Vector::zeros(N);
            for (size_t i = sum + 1; i < N; i++)
                B_k[i] = matrix[i][sum];

            B_k.scale(-T_k_inverse[0][0]);
            auto B_k_inverse = Vector(B_k);
            B_k_inverse.scale(-1.0);*/

            // more complex case
            auto b_height = N - sum - 1;
            auto B_k_1_inverse = Vector::zeros(N);
            auto B_k_2_inverse = Vector::zeros(N);
            auto B_k_1 = Vector::zeros(N);
            auto B_k_2 = Vector::zeros(N);

            for (size_t i = sum + 2; i < N; i++) {
                B_k_1_inverse[i] = matrix[i][sum];
                B_k_2_inverse[i] = matrix[i][sum + 1];
            }
            // Multiply B_k * T_k_inverse
            for (size_t i = sum + 2; i < N; i++) {
                B_k_1[i] = -B_k_1_inverse[i] * T_k_inverse[0][0] - B_k_2_inverse[i] * T_k_inverse[1][0];
                B_k_2[i] = -B_k_1_inverse[i] * T_k_inverse[0][1] - B_k_2_inverse[i] * T_k_inverse[1][1];
                B_k_1_inverse[i] = -B_k_1[i];
                B_k_2_inverse[i] = -B_k_2[i];
            }
            //print_matrix(matrix.matrix, N, "before left-multiplied matrix");
            // multiply by left
            double backed_row_1[N];
            double backed_row_2[N];
            long double backed_long_1[N];
            long double backed_long_2[N];
            for (size_t i = 0; i < N; i++){
                backed_long_1[i] = matrix[sum][i];
                backed_long_2[i] = matrix[sum + 1][i];
            }
            memcpy(backed_row_1, matrix[sum], sizeof(double) * N);
            memcpy(backed_row_2, matrix[sum + 1], sizeof(double) * N);
            for (size_t i = sum + 2; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    if (j == sum) {
                        matrix[i][j] = matrix[i][j + 1] = 0.0;
                        j++;
                        continue;
                    }
                    long double precision_saver = static_cast<long double>(matrix[i][j]);
                    precision_saver += static_cast<long double>(B_k_1[i]) * backed_long_1[j] +
                        static_cast<long double>(B_k_2[i]) * backed_long_2[j];
                    matrix[i][j] = static_cast<double>(precision_saver);
                    //matrix[i][j] += B_k_1[i] * backed_row_1[j] + B_k_2[i] * backed_row_2[j];
                    #ifdef NORMALIZE_WITH_TOLERANCE
                    matrix[i][j] = normalize(matrix[i][j]);
                    #endif
                }
            }
            //print_matrix(matrix.matrix, N, "left-multiplied matrix");

            for (size_t i = 0; i < N; i++) {
                backed_row_1[i] = matrix[i][sum];
                backed_row_2[i] = matrix[i][sum + 1];
                backed_long_1[i] = static_cast<long double>(matrix[i][sum]);
                backed_long_2[i] = static_cast<long double>(matrix[i][sum + 1]);
            }
            memset((matrix[sum] + sum + 2), 0, sizeof(double) * (N - sum - 1));
            memset((matrix[sum + 1] + sum + 2), 0, sizeof(double) * (N - sum - 1));
            for (size_t i = sum + 2; i < N; i++) {
                for (size_t j = sum + 2; j < N; j++) {
                    long double precision_saver = static_cast<long double>(matrix[i][j]);
                    precision_saver += backed_long_1[i] * static_cast<long double>(B_k_1[j]) + 
                            backed_long_2[i] * static_cast<long double>(B_k_2[j]);
                    matrix[i][j] = static_cast<double>(precision_saver);
                    //matrix[i][j] += backed_row_1[i] * B_k_1[j] + backed_row_2[i] * B_k_2[j];
                    #ifdef NORMALIZE_WITH_TOLERANCE
                    matrix[i][j] = normalize(matrix[i][j]);
                    #endif
                }
            }
            /*for (size_t i = 0; i < N; i++) {
                backed_row_1[i] = 0;
                backed_row_2[i] = 0;
            }
            for (size_t i = 0; i < N; i++) {
                for (size_t j = sum + 1; j < N; j++) {
                    backed_row_1[i] += matrix[i][j] * B_k_1_inverse[j];
                    backed_row_2[i] += matrix[i][j] * B_k_2_inverse[j];
                }
            }
            for (size_t i = 0; i < N; i++) {
                matrix[i][sum] += backed_row_1[i];
                matrix[i][sum + 1] += backed_row_2[i];
            }*/
            // and PL

            //PL.permute_columns(permutation_T);
            // multiply by right
            for (size_t i = 0; i < N; i++) {
                backed_row_1[i] = 0.0;
                backed_row_2[i] = 0.0;
            }
            for (size_t i = 0; i < N; i++) {
                for (size_t j = sum + 2; j < N; j++) {
                    backed_row_1[i] += PL[i][j] * B_k_1_inverse[j];
                    backed_row_2[i] += PL[i][j] * B_k_2_inverse[j];
                }
            }
            for (size_t i = 0; i < N; i++) {
                PL[i][sum] += backed_row_1[i];
                PL[i][sum + 1] += backed_row_2[i];
            }

            /*for (size_t i = 0; i < N; i++) {
                for (size_t j = sum + 2; j < N; j++) {
                    PL[i][j] += backed_row_1[i] * B_k_1_inverse[j] + backed_row_2[i] * B_k_2_inverse[j];
                    #ifdef NORMALIZE_WITH_TOLERANCE
                    PL[i][j] = normalize(PL[i][j]);
                    #endif
                }
            }*/
            sum += 2;
            /*for (int ttt = 0; ttt < 3; ttt++) print_straight_line(50);
            std::cout << "sum: " << sum << std::endl;
            print_matrix(input_matrix, N, "tridiagonal_matrix");
            print_matrix(pl_factor, N, "PL matrix");
            double *check_matrix = matrix_conjugation(input_matrix, pl_factor, N);
            Matrix check_matrix_wrapper = Matrix(check_matrix, N);
            print_matrix(check_matrix, N, "check_matrix");
            Matrix hilb = Matrix(hilbert_matrix(N), N);
            print_matrix(hilb.matrix, N, "hilbert_matrix");
            Matrix delta = check_matrix_wrapper - hilb;
            print_matrix(delta.matrix, N, "delta_matrix");
            std::cout << "Frobenius norm: " << (check_matrix_wrapper - hilb).frobenius_norm() << std::endl;
            std::cout << "Row norm: " << (check_matrix_wrapper - hilb).norm() << std::endl;
            check_matrix_wrapper.reject();
            for (int ttt = 0; ttt < 3; ttt++) print_straight_line(50);*/
            //print_matrix(matrix, "tridiagonal matrix");
            //print_matrix(PL, "PL");
        }
    }
    //print_matrix(matrix, "tridiagonal matrix");
    //print_matrix(PL, "PL matrix");
    matrix.reject();
    PL.reject();
}

std::vector<int> distinct_permutation_and_lower_triangular(double *PL, size_t N) {
    Matrix PL_M = Matrix(PL, N);
    auto permutation = std::vector<int>(N);
    auto permutation_inverted = std::vector<int>(N);
    for (size_t i = 0; i < N; i++) {
        auto real_index = 0;
        for (int j = N-1; j >= 0; j--) {
            if (fabs(PL_M[i][j] - 1.0) < TOLERANCE) {
                real_index = j;
                //std::cout << i << ", " <<  real_index << std::endl;
                break;
            }
        }
        permutation[i] = real_index;
        permutation_inverted[real_index] = i;
    }
    PL_M.permute_rows(permutation);
    PL_M.reject();
    return permutation_inverted;
}
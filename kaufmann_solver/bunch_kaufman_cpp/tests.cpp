#include "tests.h"

M_APM frobenius_norm_raw(double *matrix, size_t N) {
    char buf[256];
    M_APM norm = m_apm_init();
    M_APM summator = m_apm_init();
    M_APM sum_result = m_apm_init();
    m_apm_set_double(norm, 0.0);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            m_apm_set_double(summator, matrix[i * N + j]);
            m_apm_multiply(sum_result, summator, summator);
            m_apm_add(summator, norm, sum_result);
            m_apm_copy(norm, summator);
            //m_apm_to_string(buf, 40, norm);
            //printf("%s\n", buf);
        }
    m_apm_sqrt(summator, 50, norm);
    m_apm_free(norm);
    m_apm_free(sum_result);
    return summator;
}
double double_frobenius_norm_raw(double *matrix, size_t N) {
    M_APM norm = frobenius_norm_raw(matrix, N);
    char buf[256];
    m_apm_to_string(buf, 15, norm);
    printf("%s\n", buf);
    return atof(buf);
}

double *hilbert_matrix(size_t N) {
    double *matrix = new double[N * N];
    memset(matrix, 0, sizeof(double) * N * N);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            matrix[i * N + j] = 1.0 / (i + j + 1);
    return matrix;
}

Matrix hilbert_matrix_m(size_t N) {
    return Matrix(hilbert_matrix(N), N);
}

void print_matrix(double *matrix, size_t N, const char *matrix_name) {
    print_straight_line(STRAIGHT);
    std::cout << matrix_name << std::endl;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            std::cout << std::setprecision(5) << std::setw(12) << matrix[i * N + j];
        }
        std::cout << std::endl;
    }
    print_straight_line(STRAIGHT);
}

void print_matrix_m(Matrix &matrix, const char *matrix_name) {
    print_matrix(matrix.matrix, matrix.dim(), matrix_name);
}

void print_straight_line(int length) {
    char line[length + 1];
    memset(line, '-', length);
    line[length] = 0;
    std::cout << line << std::endl;
}

void bunch_kaufmann_test(size_t min_size, size_t max_size, const char *filename) {
    std::fstream fs;
    fs.open (filename, std::fstream::in | std::fstream::out | std::fstream::trunc);
    fs << "N\ttime_ms\tfrobenius_norm\trow_norm\n";

    for (size_t N = min_size; N <= max_size; N++) {
        double *m = hilbert_matrix(N);
        double *PL = identity_matrix(N);
        auto start_time = std::chrono::high_resolution_clock::now();
        bunch_kaufman(m, PL, N);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto time_delta = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        double *check_matrix = matrix_conjugation(m, PL, N);

        Matrix check_matrix_wrapper = Matrix(check_matrix, N);
        //print_matrix(check_matrix, N, "check_matrix");
        Matrix hilb = Matrix(hilbert_matrix(N), N);

        fs << N << "\t"
                << time_delta << "\t"
                << (check_matrix_wrapper - hilb).frobenius_norm() << "\t"
                << (check_matrix_wrapper - hilb).norm() << "\n";
        check_matrix_wrapper.reject();
    }
    fs.close();
}

void bunch_kaufmann_full_test(size_t min_size, size_t max_size, const char *filename) {
    std::fstream fs;
    fs.open (filename, std::fstream::in | std::fstream::out | std::fstream::trunc);
    fs << "N\ttime_ms\tfrobenius_norm\trow_norm\n";

    for (size_t N = min_size; N <= max_size; N++) {
        double *m = hilbert_matrix(N);
        double *PL = identity_matrix(N);
        auto start_time = std::chrono::high_resolution_clock::now();
        bunch_kaufman(m, PL, N);
        auto permutation = distinct_permutation_and_lower_triangular(PL, N);
        auto permutation_t = permutation_transpose(permutation);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto time_delta = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

        Matrix PL_wrapper = Matrix(PL, N);
        PL_wrapper.permute_rows(permutation);
        double *check_matrix = matrix_conjugation(m, PL, N);

        Matrix check_matrix_wrapper = Matrix(check_matrix, N);
        //print_matrix(check_matrix, N, "check_matrix");
        Matrix hilb = Matrix(hilbert_matrix(N), N);

        fs << N << "\t"
                << time_delta << "\t"
                << (check_matrix_wrapper - hilb).frobenius_norm() << "\t"
                << (check_matrix_wrapper - hilb).norm() << "\n";
        PL_wrapper.reject();
        check_matrix_wrapper.reject();
    }
    fs.close();
}
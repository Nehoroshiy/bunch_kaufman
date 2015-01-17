#include "vector_kaufman.h"
#include "matrix_kaufman.h"
#include "tests.h"

double *zero_matrix(size_t N) {
    double *matrix = new double[N * N];
    memset(matrix, 0, N * N * sizeof(double));
    return matrix;
}

double *identity_matrix(size_t N) {
    double *matrix = zero_matrix(N);
    for (size_t i = 0; i < N; i++)
        matrix[i * N + i] = 1.0;
    return matrix;
}

double *copy_matrix(double *A, size_t N) {
    double *matrix = new double[N * N];
    memcpy(matrix, A, N * N * sizeof(double));
    return matrix;
}

void matrix_multiplication(double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, size_t N) {
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < N; k++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
    //std::cout << "matrix_multiplication:" << std::endl;
    //print_matrix(A, N, "A");
    //print_matrix(B, N, "B");
    //print_matrix(C, N, "C");
}

void matrix_trans_multiplication(double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, size_t N) {
    for (size_t i = 0; i < N; i++) {
        double *row_a = A + i * N;
        double *row_c = C + i * N;
        double temp[N];
        memset(temp, 0, sizeof(double) * N);
        for (size_t j = 0; j < N; j++) {
            double *row_b = B + j * N;
            for (size_t k = 0; k < N; k++)
                temp[j] += row_a[k] * row_b[k];
        }
        //std::cout << "temp[" << i << "]:" << std::endl;
        //for (int t = 0; t < N; t++)
        //    std::cout << std::setw(5) << temp[t];
        //std::cout << std::endl;
        memcpy(row_c, temp, sizeof(double) * N);
    }
    //std::cout << "matrix_trans_multiplication:" << std::endl;
    //print_matrix(A, N, "A");
    //print_matrix(B, N, "B");
    //print_matrix(C, N, "C");
}

double *matrix_transpose(double *A, size_t N) {
    double *transp = new double[N * N];
    memset(transp, 0, sizeof(double) * N * N);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            transp[i * N + j] = A[j * N + i];
    return transp;
}

double *matrix_conjugation(double *A, double *B, size_t N) {
    double *BA = new double[N * N];
    memset(BA, 0, sizeof(double) * N * N);
    matrix_multiplication(B, A, BA, N);
    double *CONJ = new double[N * N];
    memset(CONJ, 0, sizeof(double) * N * N);
    matrix_trans_multiplication(BA, B, CONJ, N);
    delete [] BA;
    return CONJ;
}


Matrix::Matrix(size_t N) {
    matrix = new double[N * N];
    _dim = N;
    memset(matrix, 0, sizeof(double) * N * N);
}

Matrix::Matrix(const Matrix &m) {
	auto N = m.dim();
	matrix = new double[N * N];
    _dim = N;
    memcpy(matrix, m.matrix, sizeof(double) * N * N);
}

Matrix::Matrix(double *data, size_t N) {
    matrix = data;
    _dim = N;
}

double *Matrix::reject() {
    double *saved = matrix;
    matrix = nullptr;
    _dim = 0;
    return saved;
}

Matrix::~Matrix() {
    if (matrix) delete [] matrix;
}

Matrix Matrix::transpose() {
    auto N = _dim;
    auto transp = Matrix::zeros(N);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            transp[i][j] = (*this)[j][i];
    return transp;
}

double *Matrix::operator [](size_t i) {
	return matrix + _dim * i;
}

int Matrix::dim() const {
	return (int)_dim;
}

Matrix Matrix::operator *(Matrix &other) {
    auto N = this->dim();
    auto C = Matrix::zeros(N);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            for (size_t k = 0; k < N; k++)
                C[i][j] += (*this)[i][k] * other[k][j];
    return C;
}

Vector Matrix::operator *(Vector &v) {
    auto N = this->dim();
    auto result = Vector::zeros(N);

    for (size_t i = 0; i < N; i++) {
        long double temp = 0.0;
        for (size_t j = 0; j < N; j++) {
            temp += static_cast<long double>((*this)[i][j]) * static_cast<long double>(v[j]);
        }
        result[i] = static_cast<double>(temp);
    }
    return result;
}

Matrix Matrix::operator -(Matrix &m) {
    auto N = this->dim();
    auto C = Matrix(*this);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            C[i][j] -= m[i][j];
    return C;
}

std::pair<double, int> Matrix::max_in_row(int row_num, int start, int end, bool non_diagonal) {
    if (end == 0) end = _dim;
    auto max = -fabs((*this)[row_num][start]);
    auto idx = start;
    for (size_t i = start + 1; i < end; i++) {
        if (non_diagonal && row_num == i) continue;
        if (fabs((*this)[row_num][i]) > max) {
            idx = i;
            max = (*this)[row_num][i];
        }
    }
    return std::pair<double, int>(max, idx);
}

std::pair<double, int> Matrix::max_in_column(int column_num, int start, int end, bool non_diagonal) {
    if (end == 0) end = _dim;
    auto max = fabs((*this)[start][column_num]);
    auto idx = start;
    for (size_t i = start + 1; i < end; i++) {
        if (non_diagonal && column_num == i) continue;
        if (fabs((*this)[i][column_num]) > max) {
            idx = i;
            max = fabs((*this)[i][column_num]);
        }
    }
    return std::pair<double, int>(max, idx);
}

std::pair<double, int> Matrix::max_in_diagonal(int start, int end) {
    if (end == 0) end = _dim;
    auto max = fabs((*this)[start][start]);
    auto idx = start;
    for (size_t i = start + 1; i < end; i++) {
        if (fabs((*this)[i][i]) > max) {
            idx = i;
            max = fabs((*this)[i][i]);
        }
    }
    return std::pair<double, int>(max, idx);
}

void Matrix::exchange_rows(int s, int t) {
    if (s == t) return;
    double temp[_dim];
    memcpy(temp, (*this)[s], sizeof(double)*_dim);
    memcpy((*this)[s], (*this)[t], sizeof(double)*_dim);
    memcpy((*this)[t], temp, sizeof(double)*_dim);
}

void Matrix::exchange_columns(int s, int t) {
    if (s == t) return;
    double temp[_dim];
    for (size_t i = 0; i < _dim; i++) {
        temp[i] = (*this)[i][s];
    }
    for (size_t i = 0; i < _dim; i++) {
        (*this)[i][s] = (*this)[i][t];
    }
    for (size_t i = 0; i < _dim; i++) {
        (*this)[i][t] = temp[i];
    }
}

void Matrix::permute_rows(std::vector<int> &permutation) {
    auto N = permutation.size();
    auto moving = 0;
    std::vector<int> moving_from = std::vector<int>();
    std::vector<int> moving_to  = std::vector<int>();
    for (size_t i = 0; i < N; i++) {
        if (permutation[i] != i) {
            moving_from.push_back(i);
            moving_to.push_back(permutation[i]);
            moving++;
        }
    }
    if (moving == 0) return;
    if (moving == 2) {
            exchange_rows(moving_from[0], moving_to[0]);
            return;
    }
    auto temporary = new double[moving * N];
    for (size_t i = 0; i < moving; i++) {
        memcpy(temporary + N * i, (*this)[moving_from[i]], sizeof(double) * N);
    }
    for (size_t i = 0; i < moving; i++) {
        memcpy((*this)[moving_to[i]], temporary + N * i, sizeof(double) * N);
    }
    delete [] temporary;
}

void Matrix::permute_columns(std::vector<int> &permutation) {
    auto N = permutation.size();
    auto moving = 0;
    std::vector<int> moving_from = std::vector<int>();
    std::vector<int> moving_to = std::vector<int>();
    for (size_t i = 0; i < N; i++) {
        if (permutation[i] != i) {
            moving_from.push_back(i);
            moving_to.push_back(permutation[i]);
            moving++;
        }
    }
    if (moving == 0) return;
    if (moving == 2) {
        exchange_columns(moving_from[0], moving_to[0]);
        return;
    }
    // ?????????????????????????
    double temporary[moving * N];
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < moving; j++) {
            temporary[N * j + i] = (*this)[i][moving_from[j]];
        }
    }
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < moving; j++) {
            (*this)[i][moving_to[j]] = temporary[N * j + i]; 
        }
    }
    //delete [] temporary;
}

double Matrix::frobenius_norm() {
    long double sum = 0;
    for (size_t i = 0; i < _dim; i++)
        for (size_t j = 0; j < _dim; j++)
            sum += (*this)[i][j] * (*this)[i][j];
    return sqrt((double)sum);
}

double Matrix::frobenius_norm_exact() {
    return double_frobenius_norm_raw(matrix, _dim);
}

double Matrix::norm() {
    size_t N = _dim;
    double max_row = -DBL_MAX;
    double temp;
    for (size_t i = 0; i < N; i++) {
        temp = 0.0;
        for (size_t j = 0; j < N; j++) {
            temp += fabs((*this)[i][j]);
        }
        if (temp > max_row) max_row = temp;
    }
    return max_row;
}

Matrix Matrix::I(size_t N) {
    auto mtx = Matrix(N);
    for (size_t i = 0; i < N; i++)
        mtx[i][i] = 1.0;
    return mtx;
}

Matrix Matrix::zeros(size_t N) {
    return Matrix(N);
}
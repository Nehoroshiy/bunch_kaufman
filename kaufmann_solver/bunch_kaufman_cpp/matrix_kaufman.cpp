#include "vector_kaufman.h"
#include "matrix_kaufman.h"

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

Matrix::~Matrix() {
    delete [] matrix;
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

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            result[i] += (*this)[i][j] * v[j];
        }
    return result;
}

Matrix Matrix::operator -(Matrix &m) {
    auto N = this->dim();
    auto C = Matrix(*this);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++)
            C[i][j] += m[i][j];
    return C;
}

std::pair<double, int> Matrix::max_in_row(int row_num, int start, int end, bool non_diagonal) {
    if (end == 0) end = _dim;
    auto max = -DBL_MAX;
    auto idx = 0;
    for (size_t i = start; i < end; i++) {
        if (non_diagonal && row_num == i) continue;
        if ((*this)[row_num][i] > max) {
            idx = i;
            max = (*this)[row_num][i];
        }
    }
    return std::pair<double, int>(max, idx);
}

std::pair<double, int> Matrix::max_in_column(int column_num, int start, int end, bool non_diagonal) {
    if (end == 0) end = _dim;
    auto max = -DBL_MAX;
    auto idx = 0;
    for (size_t i = start; i < end; i++) {
        if (non_diagonal && column_num == i) continue;
        if ((*this)[i][column_num] > max) {
            idx = i;
            max = (*this)[i][column_num];
        }
    }
    return std::pair<double, int>(max, idx);
}

std::pair<double, int> Matrix::max_in_diagonal(int start, int end) {
    if (end == 0) end = _dim;
    auto max = -DBL_MAX;
    auto idx = 0;
    for (size_t i = start; i < end; i++) {
        if ((*this)[i][i] > max) {
            idx = i;
            max = (*this)[i][i];
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
        temp[i] = (*this)[s][i];
    }
    for (size_t i = 0; i < _dim; i++) {
        (*this)[s][i] = (*this)[t][i];
    }
    for (size_t i = 0; i < _dim; i++) {
        (*this)[t][i] = temp[i];
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

Matrix Matrix::I(size_t N) {
    auto mtx = Matrix(N);
    for (size_t i = 0; i < N; i++)
        mtx[i][i] = 1.0;
    return mtx;
}

Matrix Matrix::zeros(size_t N) {
    return Matrix(N);
}
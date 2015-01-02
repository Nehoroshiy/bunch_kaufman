#ifndef MATRIX_KAUFMAN_H
#define MATRIX_KAUFMAN_H

#include <cstring>
#include <utility>
#include <float.h>
#include <vector>
#include <math.h>
//#include "vector_kaufman.h"

class Vector;

class Matrix {
private:
    size_t _dim;
    double *matrix;
public:
    Matrix();
    explicit Matrix(size_t N);
    Matrix(const Matrix &m);
    ~Matrix();

    Matrix transpose();
    double *operator [](size_t i);
    Matrix operator *(Matrix &m);
    Matrix operator -(Matrix &m);
    Vector operator *(Vector &v);
    std::pair<double, int> max_in_row(int row_num, int start = 0, int end = 0, bool non_diagonal = false);
    std::pair<double, int> max_in_column(int column_num, int start = 0, int end = 0, bool non_diagonal = false);
    std::pair<double, int> max_in_diagonal(int start = 0, int end = 0);
    void exchange_rows(int s, int t);
    void exchange_columns(int s, int t);
    void permute_rows(std::vector<int> &permutation);
    void permute_columns(std::vector<int> &permutation);

    double frobenius_norm();

    int dim() const;
    static Matrix I(size_t N);
    static Matrix zeros(size_t M);
};

#endif
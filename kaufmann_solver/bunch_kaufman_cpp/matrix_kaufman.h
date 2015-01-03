#ifndef MATRIX_KAUFMAN_H
#define MATRIX_KAUFMAN_H

#include <stdio.h>
#include <cstring>
#include <utility>
#include <float.h>
#include <vector>
#include <math.h>
//#include "vector_kaufman.h"

class Vector;

void matrix_multiplication(double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, size_t N);

void matrix_trans_multiplication(double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, size_t N);

double *matrix_transpose(double *A, size_t N);

double *matrix_conjugation(double *A, double *B, size_t N);

class Matrix {
public:
    size_t _dim;
    double *matrix;
    Matrix();
    explicit Matrix(size_t N);
    Matrix(const Matrix &m);
    Matrix(double *data, size_t N);
    void reject();
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
    double frobenius_norm_exact();
    double norm();

    int dim() const;
    static Matrix I(size_t N);
    static Matrix zeros(size_t M);
};

#endif
#ifndef TESTS_H
#define TESTS_H

#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <string>
#include "matrix_kaufman.h"
#include "m_apm.h"

#define STRAIGHT 80

template<typename T>
void print_vector(std::vector<T> &v) {
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << std::setprecision(5) << std::setw(5) << v[i];
    }
    std::cout << std::endl;
}


M_APM frobenius_norm_raw(double *matrix, size_t N);
double double_frobenius_norm_raw(double *matrix, size_t N);
double *hilbert_matrix(size_t N);
double *identity_matrix(size_t N);
Matrix hilbert_matrix_m(size_t N);
void print_matrix(double *matrix, size_t N, const char *matrix_name);
void print_matrix_m(Matrix &matrix, const char *matrix_name);
void print_straight_line(int length);

#endif
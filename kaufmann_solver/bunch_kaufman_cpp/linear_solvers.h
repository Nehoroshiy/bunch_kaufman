#ifndef LINEAR_SOLVERS_H
#define LINEAR_SOLVERS_H

#include "matrix_kaufman.h"
#include "bunch_kaufman.h"
#include <cassert>

#define EPS 1e-9

void solve_tridiagonal(double* a, double* b, double* c, double* d, int n);

void solve_upper_triangular(double *A, double *b, size_t N);

void solve_lower_triangular(double *A, double *b, size_t N);

void bunch_kaufman_solve(double *A, double *b, size_t N);

void bunch_kaufman_solve_without_refinement(std::vector<int> P,
    double *L,
    double *tri_a,
    double *tri_b,
    double *tri_c,
    double *L_STAR,
    std::vector<int> P_T,
    double *b,
    size_t N);


#endif
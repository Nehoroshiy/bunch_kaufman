#ifndef BUNCH_KAUFMAN_H
#define BUNCH_KAUFMAN_H

#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <utility>
#include <vector>
#include "vector_kaufman.h"
#include "matrix_kaufman.h"
#include "tests.h"

#define TOLERANCE 1e-13L

//#define NORMALIZE_WITH_TOLERANCE

double normalize(double val);

std::vector<int> permute_ident(size_t N);

std::vector<int> permutation_transpose(std::vector<int> &permutation);

std::vector<int> compose_permutations(std::vector<int> &outer, std::vector<int> &inner);

Matrix inverse_1_2(Matrix &small_matrix);

void bunch_kaufman(double *input_matrix, double *pl_factor, size_t N, double alpha=(1.0 + sqrt(17)) / 8);

std::vector<int> distinct_permutation_and_lower_triangular(double *PL, size_t N);

#endif
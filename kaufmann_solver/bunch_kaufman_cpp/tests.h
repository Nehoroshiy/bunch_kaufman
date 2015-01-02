#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <string>
#include "matrix_kaufman.h"

#define STRAIGHT 80

Matrix hilbert_matrix(size_t N);
void print_matrix(Matrix &matrix, const char *matrix_name);
void print_straight_line(int length);

Matrix hilbert_matrix(size_t N) {
	auto matrix = Matrix::zeros(N);
	for (size_t i = 0; i < N; i++)
		for (size_t j = 0; j < N; j++)
			matrix[i][j] = 1.0 / (i + j + 1);
	return matrix;
}

template<typename T>
void print_vector(std::vector<T> &v) {
	for (size_t i = 0; i < v.size(); i++) {
		std::cout << std::setprecision(5) << std::setw(5) << v[i];
	}
	std::cout << std::endl;
}

void print_matrix(Matrix &matrix, const char *matrix_name) {
	print_straight_line(STRAIGHT);
	std::cout << matrix_name << std::endl;
	for (size_t i = 0; i < matrix.dim(); i++) {
		for (size_t j = 0; j < matrix.dim(); j++) {
			std::cout << std::setprecision(5) << std::setw(15) << matrix[i][j];
		}
		std::cout << std::endl;
	}
	print_straight_line(STRAIGHT);
}

void print_straight_line(int length) {
	char line[length + 1];
	memset(line, '-', length);
	line[length] = 0;
	std::cout << line << std::endl;
}
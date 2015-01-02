#ifndef VECTOR_KAUFMAN_H
#define VECTOR_KAUFMAN_H

class Matrix;

#include <cstring>
#include <math.h>
//#include "matrix_kaufman.h"

class Vector {
private:
    size_t _dim;
    double *data;
public:
    Vector();
    explicit Vector(size_t N);
    Vector(const Vector &v);
    ~Vector();

    int dim() const;

    double &operator [](size_t i);
    Vector operator *(Matrix &m);
    double operator *(Vector &v);
    void scale(double alpha);
    double euclid_norm();

    static Vector I(size_t N);
    static Vector zeros(size_t N);
};

#endif
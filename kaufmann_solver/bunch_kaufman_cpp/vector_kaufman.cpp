#include "vector_kaufman.h"
#include "matrix_kaufman.h"

Vector::Vector(size_t N) {
    data = new double[N];
    _dim = N;
    memset(data, 0, sizeof(double) * N);
}

Vector::Vector(const Vector &v) {
    data = new double[v.dim()];
    _dim = v.dim();
    memcpy(data, v.data, sizeof(double) * v.dim());
}

Vector::Vector(double *some_data, size_t N) {
    data = some_data;
    _dim = N;
}

double *Vector::reject() {
    double *saved = data;
    data = nullptr;
    _dim = 0;
    return saved;
}

Vector::~Vector() {
    if (data) delete [] data;
}

double &Vector::operator [](size_t i) const {
    return data[i];
}

int Vector::dim() const {
    return (int)_dim;
}

Vector Vector::operator -() const {
    Vector result = Vector(*this);
    for (size_t i = 0; i < _dim; i++)
        result[i] = -(*this)[i];
    return result;
}
Vector Vector::operator +(Vector &v) const {
    Vector result = Vector(*this);
    for (size_t i = 0; i < _dim; i++)
        result[i] += v[i];
    return result;
}
Vector Vector::operator -(Vector &v) const {
    Vector result = Vector(*this);
    for (size_t i = 0; i < _dim; i++)
        result[i] -= v[i];
    return result;
}

Vector Vector::operator -(Vector &&v) const {
    Vector result = Vector(*this);
    for (size_t i = 0; i < _dim; i++)
        result[i] -= v[i];
    return result;
}

void Vector::operator =(Vector &v) {
    //assert(_dim == v._dim);
    memcpy(data, v.data, sizeof(double) * v.dim());
}

void Vector::operator =(Vector &&v) {
    //assert(_dim == v._dim);
    data = v.data;
    _dim = v._dim;
    v.data = nullptr;
    v._dim = 0;
}

Vector Vector::operator *(Matrix &m) {
    auto N = this->dim();
    auto result = Vector::zeros(N);
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < N; j++) {
            result[j] += (*this)[j] * m[i][j];
        }
    return result;
}

double Vector::operator *(Vector &v) {
    auto N = this->dim();
    double result = 0;
    for (size_t i = 0; i < N; i++) {
        result += (*this)[i] * v[i];
    }
    return result;
}

void Vector::scale(double alpha) {
    for (size_t i = 0; i < _dim; i++)
        data[i] *= alpha;
}

double Vector::euclid_norm() {
    double sum = 0;
    for (size_t i = 0; i < _dim; i++)
        sum += data[i]*data[i];
    return sqrt(sum);
}

Vector Vector::I(size_t N) {
    auto vector = Vector(N);
    for (size_t i = 0; i < N; i++)
        vector[i] = 1.0;
    return vector;
}

Vector Vector::zeros(size_t N) {
    return Vector(N);
}
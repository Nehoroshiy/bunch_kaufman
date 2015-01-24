#include "linear_solvers.h"

void solve_tridiagonal(double* a, double* b, double* c, double* d, int n) {
    /*
    // n is the number of unknowns

    |b0 c0 0 ||x0| |d0|
    |a1 b1 c1||x1|=|d1|
    |0  a2 b2||x2| |d2|

    1st iteration: b0x0 + c0x1 = d0 -> x0 + (c0/b0)x1 = d0/b0 ->

        x0 + g0x1 = r0               where g0 = c0/b0        , r0 = d0/b0

    2nd iteration:     | a1x0 + b1x1   + c1x2 = d1
        from 1st it.: -| a1x0 + a1g0x1        = a1r0
                    -----------------------------
                          (b1 - a1g0)x1 + c1x2 = d1 - a1r0

        x1 + g1x2 = r1               where g1=c1/(b1 - a1g0) , r1 = (d1 - a1r0)/(b1 - a1g0)

    3rd iteration:      | a2x1 + b2x2   = d2
        from 2st it. : -| a2x1 + a2g1x2 = a2r2
                       -----------------------
                       (b2 - a2g1)x2 = d2 - a2r2
        x2 = r2                      where                     r2 = (d2 - a2r2)/(b2 - a2g1)
    Finally we have a triangular matrix:
    |1  g0 0 ||x0| |r0|
    |0  1  g1||x1|=|r1|
    |0  0  1 ||x2| |r2|

    Condition: ||bi|| > ||ai|| + ||ci||

    in this version the c matrix reused instead of g
    and             the d matrix reused instead of r and x matrices to report results
    Written by Keivan Moradi, 2014
    */
    n--; // since we start from x0 (not x1)
    c[0] /= b[0];
    d[0] /= b[0];

    for (int i = 1; i < n; i++) {
        c[i] /= b[i] - a[i]*c[i-1];
        d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1]);
    }

    d[n] = (d[n] - a[n]*d[n-1]) / (b[n] - a[n]*c[n-1]);

    for (int i = n; i-- > 0;) {
        d[i] -= c[i]*d[i+1];
    }
}

void solve_upper_triangular(double *A, double *b, size_t N) {
    long double b_cpy[N];
    for (size_t i = 0; i < N; i++)
        b_cpy[i] = static_cast<long double>(b[i]);
    //double *x = new double[N];
    for (int i = N - 1; i >= 0; i--) {
        //x[i] = static_cast<double>(b_cpy[i]);
        b[i] = static_cast<double>(b_cpy[i]);
        for (int j = 0; j < i; j++) {
            b_cpy[j] -= static_cast<long double>(A[j * N + i]) * static_cast<long double>(b[i]);
        }
    }
}

void solve_lower_triangular(double *A, double *b, size_t N) {
    long double b_cpy[N];
    for (size_t i = 0; i < N; i++)
        b_cpy[i] = static_cast<long double>(b[i]);
    //double *x = new double[N];
    for (int i = 0; i < N; i++) {
        //x[i] = static_cast<double>(b_cpy[i]);
        b[i] = static_cast<double>(b_cpy[i]);
        for (int j = i + 1; j < N; j++) {
            b_cpy[j] -= static_cast<long double>(A[j * N + i]) * static_cast<long double>(b[i]);
        }
    }
}

void bunch_kaufman_solve(double *A, double *b, size_t N) {
    // make bunch_kaufmann factorization
    double *m = copy_matrix(A, N);
    double *L = identity_matrix(N);
    bunch_kaufman(m, L, N);
    auto P = distinct_permutation_and_lower_triangular(L, N);
    auto P_T = permutation_transpose(P);
    double *L_STAR = matrix_transpose(L, N);

//---------------------------------
// Fill three diagonals of T
    double *tri_a = new double[N];
    double *tri_b = new double[N];
    double *tri_c = new double[N];
    tri_a[0] = 0; tri_b[0] = m[0 + 0]; tri_c[0] = m[0 + 1];
    for (size_t i = 1; i < N - 1; i++) {
        tri_a[i] = m[i * N + i - 1];
        tri_b[i] = m[i * N + i];
        tri_c[i] = m[i * N + i + 1];
    }
    tri_a[N - 1] = m[N * (N - 1) + N - 2];
    tri_b[N - 1] = m[N * (N - 1) + (N - 1)];
    tri_c[N - 1] = 0;
//---------------------------------
    delete [] m;

    double *origin = new double[N];
    double *residual = new double[N];
    for (size_t i = 0; i < N; i++) {
        origin[i] = b[i];
    }
    Matrix system_matrix_wrapper = Matrix(A, N);
    Vector origin_wrapper = Vector(origin, N);
    Vector residual_wrapper = Vector(residual, N);
    Vector computed_wrapper = Vector(b, N);
    bunch_kaufman_solve_without_refinement(
        P, L, tri_a, tri_b, tri_c, L_STAR, P_T, b, N);
    /*residual_wrapper = origin_wrapper - system_matrix_wrapper * computed_wrapper;
    while (residual_wrapper.euclid_norm()) {
        bunch_kaufman_solve_without_refinement(
            P, L, tri_a, tri_b, tri_c, L_STAR, P_T, residual_wrapper.data, N);
        for (size_t i = 0; i < N; i++)
            computed_wrapper[i] += residual_wrapper[i];
        residual_wrapper = origin_wrapper - system_matrix_wrapper * computed_wrapper;
    }*/

    system_matrix_wrapper.reject();
    origin_wrapper.reject();
    residual_wrapper.reject();
    computed_wrapper.reject();

    delete [] L;
    delete [] L_STAR;
    delete [] tri_a;
    delete [] tri_b;
    delete [] tri_c;
}

void bunch_kaufman_solve_without_refinement(std::vector<int> P,
        double *L,
        double *tri_a,
        double *tri_b,
        double *tri_c,
        double *L_STAR,
        std::vector<int> P_T,
        double *b,
        size_t N) {
    double b_cpy[N];
    for (size_t i = 0; i < N; i++) {
        b_cpy[P_T[i]] = b[i];
    }
    // First equation
    solve_lower_triangular(L, b_cpy, N);

    // Second equation
    solve_tridiagonal(tri_a, tri_b, tri_c, b_cpy, N);

    // Third equation
    solve_upper_triangular(L_STAR, b_cpy, N);

    // Forth equation
    for (size_t i = 0; i < N; i++) {
        b[P[i]] = b_cpy[i];
    }
    //delete [] b_cpy;
}








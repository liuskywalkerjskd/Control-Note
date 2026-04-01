// mat.hpp — Lightweight Compile-Time Matrix Library
// No heap, no STL, no exceptions. For embedded Kalman/LQR.
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include <cstring>
#include <cmath>

template<int ROWS, int COLS>
struct Mat {
    float data[ROWS][COLS] = {};

    float& operator()(int r, int c)       { return data[r][c]; }
    float  operator()(int r, int c) const { return data[r][c]; }

    static Mat zeros() { Mat m; return m; }

    static Mat identity() {
        static_assert(ROWS == COLS, "Identity requires square matrix");
        Mat m;
        for (int i = 0; i < ROWS; ++i) m.data[i][i] = 1.0f;
        return m;
    }

    Mat operator+(const Mat& rhs) const {
        Mat r;
        for (int i = 0; i < ROWS; ++i)
            for (int j = 0; j < COLS; ++j)
                r.data[i][j] = data[i][j] + rhs.data[i][j];
        return r;
    }

    Mat operator-(const Mat& rhs) const {
        Mat r;
        for (int i = 0; i < ROWS; ++i)
            for (int j = 0; j < COLS; ++j)
                r.data[i][j] = data[i][j] - rhs.data[i][j];
        return r;
    }

    template<int COLS2>
    Mat<ROWS, COLS2> operator*(const Mat<COLS, COLS2>& rhs) const {
        Mat<ROWS, COLS2> r;
        for (int i = 0; i < ROWS; ++i)
            for (int j = 0; j < COLS2; ++j)
                for (int k = 0; k < COLS; ++k)
                    r.data[i][j] += data[i][k] * rhs.data[k][j];
        return r;
    }

    Mat operator*(float s) const {
        Mat r;
        for (int i = 0; i < ROWS; ++i)
            for (int j = 0; j < COLS; ++j)
                r.data[i][j] = data[i][j] * s;
        return r;
    }

    Mat<COLS, ROWS> T() const {
        Mat<COLS, ROWS> r;
        for (int i = 0; i < ROWS; ++i)
            for (int j = 0; j < COLS; ++j)
                r.data[j][i] = data[i][j];
        return r;
    }

    Mat inverse() const;
};

// 1x1 inverse
template<> inline Mat<1,1> Mat<1,1>::inverse() const {
    Mat<1,1> r;
    r.data[0][0] = 1.0f / data[0][0];
    return r;
}

// 2x2 inverse
template<> inline Mat<2,2> Mat<2,2>::inverse() const {
    Mat<2,2> r;
    float det = data[0][0]*data[1][1] - data[0][1]*data[1][0];
    float inv_det = 1.0f / det;
    r.data[0][0] =  data[1][1] * inv_det;
    r.data[0][1] = -data[0][1] * inv_det;
    r.data[1][0] = -data[1][0] * inv_det;
    r.data[1][1] =  data[0][0] * inv_det;
    return r;
}

// 3x3 inverse
template<> inline Mat<3,3> Mat<3,3>::inverse() const {
    Mat<3,3> r;
    float det = data[0][0]*(data[1][1]*data[2][2]-data[1][2]*data[2][1])
              - data[0][1]*(data[1][0]*data[2][2]-data[1][2]*data[2][0])
              + data[0][2]*(data[1][0]*data[2][1]-data[1][1]*data[2][0]);
    float inv = 1.0f / det;
    r(0,0)=(data[1][1]*data[2][2]-data[1][2]*data[2][1])*inv;
    r(0,1)=(data[0][2]*data[2][1]-data[0][1]*data[2][2])*inv;
    r(0,2)=(data[0][1]*data[1][2]-data[0][2]*data[1][1])*inv;
    r(1,0)=(data[1][2]*data[2][0]-data[1][0]*data[2][2])*inv;
    r(1,1)=(data[0][0]*data[2][2]-data[0][2]*data[2][0])*inv;
    r(1,2)=(data[0][2]*data[1][0]-data[0][0]*data[1][2])*inv;
    r(2,0)=(data[1][0]*data[2][1]-data[1][1]*data[2][0])*inv;
    r(2,1)=(data[0][1]*data[2][0]-data[0][0]*data[2][1])*inv;
    r(2,2)=(data[0][0]*data[1][1]-data[0][1]*data[1][0])*inv;
    return r;
}

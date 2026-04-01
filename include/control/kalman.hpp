// kalman.hpp — Generic Discrete-Time Kalman Filter
// Template parameters set dimensions at compile time. No heap allocation.
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include "mat.hpp"

template<int N_STATE, int N_MEAS, int N_INPUT = 1>
class KalmanFilter {
public:
    using StateVec  = Mat<N_STATE, 1>;
    using MeasVec   = Mat<N_MEAS, 1>;
    using InputVec  = Mat<N_INPUT, 1>;
    using StateMat  = Mat<N_STATE, N_STATE>;
    using InputMat  = Mat<N_STATE, N_INPUT>;
    using MeasMat   = Mat<N_MEAS, N_STATE>;
    using GainMat   = Mat<N_STATE, N_MEAS>;
    using MeasCov   = Mat<N_MEAS, N_MEAS>;

    StateMat A;   // State transition matrix
    InputMat B;   // Input matrix
    MeasMat  C;   // Measurement matrix
    StateMat Q;   // Process noise covariance
    MeasCov  R;   // Measurement noise covariance

    StateVec x;   // State estimate
    StateMat P;   // Estimate covariance

    /// Predict step: propagate state and covariance forward
    void predict(const InputVec& u) {
        x = A * x + B * u;
        P = A * P * A.T() + Q;
    }

    /// Predict step (no input)
    void predict() {
        x = A * x;
        P = A * P * A.T() + Q;
    }

    /// Update step: incorporate measurement
    void update(const MeasVec& z) {
        // Innovation
        MeasVec y = z - C * x;
        // Innovation covariance
        MeasCov S = C * P * C.T() + R;
        // Kalman gain
        GainMat K = P * C.T() * S.inverse();
        // State update
        x = x + K * y;
        // Covariance update (Joseph form for numerical stability)
        StateMat I_KC = StateMat::identity() - K * C;
        P = I_KC * P * I_KC.T() + K * R * K.T();
    }

    /// Combined predict + update
    void step(const InputVec& u, const MeasVec& z) {
        predict(u);
        update(z);
    }
};

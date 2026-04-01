// lqr.hpp — LQR Controller (Runtime Gain Application)
// Gain K is computed offline in MATLAB/Python, applied here at runtime.
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include "mat.hpp"

template<int N_STATE, int N_INPUT>
class LQRController {
public:
    using StateVec = Mat<N_STATE, 1>;
    using InputVec = Mat<N_INPUT, 1>;
    using GainMat  = Mat<N_INPUT, N_STATE>;

    GainMat K;  // Set from MATLAB: K = lqr(A, B, Q, R)

    /// Compute optimal input: u = -K * (x - x_ref)
    InputVec compute(const StateVec& x, const StateVec& x_ref) const {
        StateVec error = x - x_ref;
        return (K * error) * (-1.0f);
    }

    /// Compute optimal input for regulation (x_ref = 0)
    InputVec compute(const StateVec& x) const {
        return (K * x) * (-1.0f);
    }
};

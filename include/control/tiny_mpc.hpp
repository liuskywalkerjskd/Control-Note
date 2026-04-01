// tiny_mpc.hpp — Riccati-based MPC for embedded systems
// Based on: TinyMPC (Nguyen et al., 2024)
// Precomputes Riccati recursion offline, solves ADMM online.
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include "mat.hpp"

template<int NX, int NU, int N_HORIZON>
class TinyMPC {
public:
    using StateVec = Mat<NX, 1>;
    using InputVec = Mat<NU, 1>;
    using StateMat = Mat<NX, NX>;
    using InputMat = Mat<NX, NU>;
    using GainMat  = Mat<NU, NX>;
    using InputCov = Mat<NU, NU>;

    // System model (discrete-time)
    StateMat Ad;
    InputMat Bd;

    // Cost matrices
    StateMat Q;          // State stage cost
    InputCov R;          // Input stage cost
    StateMat Q_terminal; // Terminal cost (set to DARE solution for stability)

    // Box constraints
    InputVec u_min, u_max;
    StateVec x_min, x_max;

    // ADMM parameters
    float rho = 1.0f;        // Penalty parameter
    int   max_iter = 10;      // ADMM iterations (5-20 typical on MCU)

    // Precomputed Riccati data
    GainMat  K_gains[N_HORIZON];
    StateMat P_mats[N_HORIZON + 1];

    /// Precompute Riccati recursion (call once at startup)
    void precompute() {
        P_mats[N_HORIZON] = Q_terminal;
        for (int k = N_HORIZON - 1; k >= 0; --k) {
            auto P_next = P_mats[k + 1];
            auto BtP = Bd.T() * P_next;
            auto S = R + BtP * Bd;
            auto S_inv = S.inverse();
            K_gains[k] = S_inv * BtP * Ad;
            auto AtP = Ad.T() * P_next;
            P_mats[k] = Q + AtP * Ad - AtP * Bd * K_gains[k];
        }
    }

    /// Solve constrained MPC. Returns optimal first input.
    InputVec solve(const StateVec& x0,
                   const StateVec x_ref[N_HORIZON]) {
        // Forward rollout with unconstrained LQR
        StateVec x_traj[N_HORIZON + 1];
        InputVec u_traj[N_HORIZON];
        x_traj[0] = x0;

        for (int k = 0; k < N_HORIZON; ++k) {
            StateVec dx = x_traj[k] - x_ref[k];
            u_traj[k] = (K_gains[k] * dx) * (-1.0f);
            clamp_vec(u_traj[k], u_min, u_max);
            x_traj[k + 1] = Ad * x_traj[k] + Bd * u_traj[k];
        }

        // ADMM for constraint refinement
        InputVec z[N_HORIZON];
        InputVec lam[N_HORIZON];
        for (int k = 0; k < N_HORIZON; ++k) {
            z[k] = u_traj[k];
            lam[k] = InputVec::zeros();
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            // Primal update (forward pass)
            x_traj[0] = x0;
            for (int k = 0; k < N_HORIZON; ++k) {
                StateVec dx = x_traj[k] - x_ref[k];
                InputVec u_unc = (K_gains[k] * dx) * (-1.0f);
                // ADMM correction toward feasible z
                u_traj[k] = u_unc + (z[k] - lam[k]) * (rho * 0.5f);
                x_traj[k + 1] = Ad * x_traj[k] + Bd * u_traj[k];
            }
            // z-update: project onto box constraints
            for (int k = 0; k < N_HORIZON; ++k) {
                InputVec v = u_traj[k] + lam[k];
                clamp_vec(v, u_min, u_max);
                z[k] = v;
            }
            // Dual update
            for (int k = 0; k < N_HORIZON; ++k) {
                lam[k] = lam[k] + u_traj[k] - z[k];
            }
        }
        return u_traj[0];
    }

    /// Solve with constant reference (regulation)
    InputVec solve(const StateVec& x0, const StateVec& x_ref_const) {
        StateVec refs[N_HORIZON];
        for (int k = 0; k < N_HORIZON; ++k) refs[k] = x_ref_const;
        return solve(x0, refs);
    }

private:
    static void clamp_vec(InputVec& v, const InputVec& lo, const InputVec& hi) {
        for (int i = 0; i < NU; ++i) {
            float val = v(i, 0);
            val = val < lo(i, 0) ? lo(i, 0) : (val > hi(i, 0) ? hi(i, 0) : val);
            v(i, 0) = val;
        }
    }
};

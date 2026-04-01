// pid_advanced.hpp — Advanced PID with integral separation, PI-D/I-PD, 2-DOF
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include <cmath>

struct AdvancedPIDConfig {
    float kp = 0.0f, ki = 0.0f, kd = 0.0f;
    float dt = 0.001f;
    float out_min = -1e6f, out_max = 1e6f;
    float integral_max = 1e6f;
    float d_filter_N = 10.0f;        // Derivative LPF coefficient
    float setpoint_weight_b = 1.0f;  // P setpoint weight (0-1)
    float setpoint_weight_c = 0.0f;  // D setpoint weight (0=PI-D, 1=PID)
    float integral_sep_eps = 0.0f;   // 0 = disabled, >0 = threshold
};

class AdvancedPID {
public:
    AdvancedPID() = default;
    explicit AdvancedPID(const AdvancedPIDConfig& c) : cfg_(c) {}
    void configure(const AdvancedPIDConfig& c) { cfg_ = c; }

    float update(float setpoint, float measurement) {
        float error = setpoint - measurement;

        // --- P with setpoint weighting (2-DOF) ---
        float p_term = cfg_.kp * (cfg_.setpoint_weight_b * setpoint
                                  - measurement);

        // --- I with integral separation ---
        if (cfg_.integral_sep_eps <= 0.0f
            || fabsf(error) < cfg_.integral_sep_eps) {
            integral_ += cfg_.ki * error * cfg_.dt;
        }
        // Anti-windup: back-calculation
        integral_ += anti_windup_;
        integral_ = clamp(integral_, -cfg_.integral_max, cfg_.integral_max);

        // --- D with setpoint weighting + filtered derivative ---
        float d_input = cfg_.setpoint_weight_c * setpoint - measurement;
        float d_raw = -(d_input - d_prev_) / cfg_.dt;
        float alpha = cfg_.dt * cfg_.d_filter_N
                    / (1.0f + cfg_.dt * cfg_.d_filter_N);
        d_filtered_ = alpha * d_raw + (1.0f - alpha) * d_filtered_;
        float d_term = cfg_.kd * d_filtered_;
        d_prev_ = d_input;

        // --- Output ---
        float raw = p_term + integral_ + d_term;
        float out = clamp(raw, cfg_.out_min, cfg_.out_max);

        // Back-calculation anti-windup
        anti_windup_ = (cfg_.ki != 0.0f)
            ? (out - raw) / cfg_.ki * cfg_.dt * 0.5f
            : 0.0f;
        return out;
    }

    void reset() {
        integral_ = 0; d_filtered_ = 0; d_prev_ = 0; anti_windup_ = 0;
    }

    float get_integral() const { return integral_; }

private:
    AdvancedPIDConfig cfg_;
    float integral_ = 0, d_filtered_ = 0, d_prev_ = 0, anti_windup_ = 0;
    static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

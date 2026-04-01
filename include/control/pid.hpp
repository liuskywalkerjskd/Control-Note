// pid.hpp — Full-featured PID Controller
// Anti-windup (back-calculation), derivative-on-measurement, filtered derivative
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include <cmath>

struct PIDConfig {
    float kp = 0.0f;
    float ki = 0.0f;
    float kd = 0.0f;
    float dt = 0.001f;         // Sampling period (s)
    float out_min = -1e6f;     // Output lower limit
    float out_max =  1e6f;     // Output upper limit
    float integral_max = 1e6f; // Integral term clamp
    float d_filter_N = 10.0f;  // Derivative filter coefficient
};

class PID {
public:
    PID() = default;
    explicit PID(const PIDConfig& cfg) : cfg_(cfg) {}

    void configure(const PIDConfig& cfg) { cfg_ = cfg; }

    /// Compute PID output. Call once per control cycle.
    float update(float setpoint, float measurement) {
        float error = setpoint - measurement;

        // --- Proportional ---
        float p_term = cfg_.kp * error;

        // --- Integral with back-calculation anti-windup ---
        integral_ += cfg_.ki * error * cfg_.dt + anti_windup_;
        integral_ = clamp(integral_, -cfg_.integral_max, cfg_.integral_max);

        // --- Derivative on measurement (no derivative kick) ---
        float d_raw = -(measurement - meas_prev_) / cfg_.dt;
        float alpha_d = cfg_.dt * cfg_.d_filter_N
                      / (1.0f + cfg_.dt * cfg_.d_filter_N);
        d_filtered_ = alpha_d * d_raw + (1.0f - alpha_d) * d_filtered_;
        float d_term = cfg_.kd * d_filtered_;

        meas_prev_ = measurement;

        // --- Total output ---
        float output_raw = p_term + integral_ + d_term;
        float output = clamp(output_raw, cfg_.out_min, cfg_.out_max);

        // --- Back-calculation anti-windup ---
        if (cfg_.ki != 0.0f) {
            anti_windup_ = (output - output_raw) * (1.0f / cfg_.ki)
                         * cfg_.dt * 0.5f;
        } else {
            anti_windup_ = 0.0f;
        }

        return output;
    }

    void reset() {
        integral_ = 0.0f;
        d_filtered_ = 0.0f;
        meas_prev_ = 0.0f;
        anti_windup_ = 0.0f;
    }

    float get_integral() const { return integral_; }

private:
    PIDConfig cfg_;
    float integral_    = 0.0f;
    float d_filtered_  = 0.0f;
    float meas_prev_   = 0.0f;
    float anti_windup_ = 0.0f;

    static float clamp(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

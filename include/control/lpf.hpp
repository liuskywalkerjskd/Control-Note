// lpf.hpp — First-order IIR Low-Pass Filter
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

class LowPassFilter {
public:
    LowPassFilter() : alpha_(1.0f), y_prev_(0.0f), initialized_(false) {}

    /// Configure with cutoff frequency (Hz) and sampling period (s)
    void configure(float cutoff_hz, float dt) {
        float rc = 1.0f / (2.0f * static_cast<float>(M_PI) * cutoff_hz);
        alpha_ = dt / (rc + dt);
    }

    /// Configure directly with smoothing factor (0 < alpha <= 1)
    void set_alpha(float alpha) { alpha_ = alpha; }

    /// Process one sample. Call this at your fixed sample rate.
    float update(float x) {
        if (!initialized_) {
            y_prev_ = x;
            initialized_ = true;
            return x;
        }
        y_prev_ = alpha_ * x + (1.0f - alpha_) * y_prev_;
        return y_prev_;
    }

    /// Reset filter state
    void reset() { initialized_ = false; y_prev_ = 0.0f; }

    /// Get current filtered value without feeding new input
    float value() const { return y_prev_; }

private:
    float alpha_;
    float y_prev_;
    bool  initialized_;
};

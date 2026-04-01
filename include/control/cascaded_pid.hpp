// cascaded_pid.hpp — Two-Loop Cascaded PID (Position + Velocity)
// Outer position loop feeds inner velocity loop, with optional feedforward.
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include "pid.hpp"

class CascadedPID {
public:
    PID outer;   // Position loop (slower)
    PID inner;   // Velocity loop (faster)

    struct Config {
        PIDConfig outer_cfg;
        PIDConfig inner_cfg;
        float ff_gain = 0.0f;  // Velocity feedforward gain
    };

    void configure(const Config& cfg) {
        outer.configure(cfg.outer_cfg);
        inner.configure(cfg.inner_cfg);
        ff_gain_ = cfg.ff_gain;
    }

    /// Full cascaded update
    /// pos_setpoint: desired position
    /// pos_measured: actual position
    /// vel_measured: actual velocity
    /// vel_feedforward: reference velocity for feedforward (optional)
    float update(float pos_setpoint, float pos_measured,
                 float vel_measured, float vel_feedforward = 0.0f) {
        float vel_cmd = outer.update(pos_setpoint, pos_measured)
                      + ff_gain_ * vel_feedforward;
        float output = inner.update(vel_cmd, vel_measured);
        return output;
    }

    void reset() { outer.reset(); inner.reset(); }

private:
    float ff_gain_ = 0.0f;
};

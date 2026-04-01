// ramp.hpp — Setpoint Ramp (LPF Soft-Start)
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include "lpf.hpp"

class SetpointRamp {
public:
    /// ramp_time: approximate time to reach 95% of a step change (seconds)
    void configure(float ramp_time, float dt) {
        float cutoff = 3.0f / (2.0f * static_cast<float>(M_PI) * ramp_time);
        filter_.configure(cutoff, dt);
    }

    float update(float raw_setpoint) {
        return filter_.update(raw_setpoint);
    }

    void reset() { filter_.reset(); }

private:
    LowPassFilter filter_;
};

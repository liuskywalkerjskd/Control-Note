// complementary_filter.hpp — IMU Tilt Estimation
// Fuses accelerometer (gravity tilt) and gyroscope (angular rate)
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include <cmath>

class ComplementaryFilter {
public:
    /// alpha: gyro trust factor (0.95-0.99 typical)
    void configure(float alpha, float dt) {
        alpha_ = alpha;
        dt_ = dt;
    }

    /// accel_angle: tilt from accelerometer (rad)
    /// gyro_rate:   angular rate from gyro (rad/s)
    /// returns:     fused angle estimate (rad)
    float update(float accel_angle, float gyro_rate) {
        angle_ = alpha_ * (angle_ + gyro_rate * dt_)
               + (1.0f - alpha_) * accel_angle;
        return angle_;
    }

    void reset(float initial_angle = 0.0f) { angle_ = initial_angle; }
    float angle() const { return angle_; }

private:
    float alpha_ = 0.98f;
    float dt_    = 0.001f;
    float angle_ = 0.0f;
};

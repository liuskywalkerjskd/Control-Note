// ekf_imu.hpp — 6-DOF IMU Pose Estimation via Extended Kalman Filter
// State: [roll, pitch, yaw, bias_gx, bias_gy, bias_gz]
// Measurement: accelerometer → roll, pitch (yaw unobservable from accel alone)
// Part of: A Practical Guide to Control Theory
// License: MIT
#pragma once
#include "mat.hpp"
#include <cmath>

class EKF_IMU {
public:
    static constexpr int NX = 6;
    static constexpr int NZ = 2;

    using State  = Mat<NX, 1>;
    using Cov    = Mat<NX, NX>;
    using MeasV  = Mat<NZ, 1>;
    using MeasC  = Mat<NZ, NZ>;
    using KGain  = Mat<NX, NZ>;
    using Hmtx   = Mat<NZ, NX>;

    State x;
    Cov   P;
    Cov   Q;
    MeasC R;
    float dt = 0.001f;

    void init() {
        x = State::zeros();
        P = Cov::identity() * 0.1f;
        Q = Cov::zeros();
        Q(0,0)=0.001f; Q(1,1)=0.001f; Q(2,2)=0.001f;
        Q(3,3)=0.0001f; Q(4,4)=0.0001f; Q(5,5)=0.0001f;
        R(0,0) = 0.15f; R(1,1) = 0.15f;
    }

    void update(float gx, float gy, float gz,
                float ax, float ay, float az) {
        float phi = x(0,0), theta = x(1,0);
        float wx = gx - x(3,0), wy = gy - x(4,0), wz = gz - x(5,0);
        float sp = sinf(phi), cp = cosf(phi);
        float tt = tanf(theta), ct = cosf(theta);

        float phi_dot   = wx + sp*tt*wy + cp*tt*wz;
        float theta_dot = cp*wy - sp*wz;
        float psi_dot   = (sp/ct)*wy + (cp/ct)*wz;

        State xp = x;
        xp(0,0) += phi_dot * dt;
        xp(1,0) += theta_dot * dt;
        xp(2,0) += psi_dot * dt;

        Cov F = Cov::identity();
        F(0,0) += (cp*tt*wy - sp*tt*wz)*dt;
        F(0,1)  = (sp/(ct*ct)*wy + cp/(ct*ct)*wz)*dt;
        F(0,3) = -dt;  F(0,4) = -sp*tt*dt;  F(0,5) = -cp*tt*dt;
        F(1,0)  = (-sp*wy - cp*wz)*dt;
        F(1,4) = -cp*dt;  F(1,5) = sp*dt;
        F(2,0)  = (cp/ct*wy - sp/ct*wz)*dt;
        float st = sinf(theta);
        F(2,1)  = (sp*st/(ct*ct)*wy + cp*st/(ct*ct)*wz)*dt;
        F(2,4) = -sp/ct*dt;  F(2,5) = -cp/ct*dt;

        Cov Pp = F * P * F.T() + Q;

        float z_roll  = atan2f(ay, az);
        float z_pitch = atan2f(-ax, sqrtf(ay*ay + az*az));

        MeasV innov;
        innov(0,0) = wrap_pi(z_roll  - xp(0,0));
        innov(1,0) = wrap_pi(z_pitch - xp(1,0));

        Hmtx H = Hmtx::zeros();
        H(0,0) = 1.0f; H(1,1) = 1.0f;

        MeasC S = H * Pp * H.T() + R;
        KGain Kk = Pp * H.T() * S.inverse();

        x = xp + Kk * innov;
        P = (Cov::identity() - Kk * H) * Pp;
    }

    float roll()   const { return x(0,0); }
    float pitch()  const { return x(1,0); }
    float yaw()    const { return x(2,0); }
    float bias_x() const { return x(3,0); }
    float bias_y() const { return x(4,0); }
    float bias_z() const { return x(5,0); }

private:
    static float wrap_pi(float a) {
        while (a >  3.14159265f) a -= 6.28318530f;
        while (a < -3.14159265f) a += 6.28318530f;
        return a;
    }
};

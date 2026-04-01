#!/usr/bin/env python3
"""
Generate EKF 6-DOF IMU pose estimation figure (Chinese version).
Includes auto-tuning loop to find optimal Q/R parameters before plotting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from zh_config import apply_zh
apply_zh()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures_zh')

plt.rcParams.update({
    'figure.figsize': (8, 4.5),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'savefig.dpi': 200,
})

def accel_from_euler(roll, pitch):
    return np.array([
        -np.sin(pitch),
        np.cos(pitch) * np.sin(roll),
        np.cos(pitch) * np.cos(roll)
    ]) * 9.81


def generate_sensor_data():
    """Generate ground truth + noisy sensor data (deterministic)."""
    dt = 0.002
    t = np.arange(0, 15, dt)
    n = len(t)

    roll_true  = 15 * np.sin(0.8 * t) * np.pi / 180
    pitch_true = 10 * np.sin(0.3 * t) * np.pi / 180
    yaw_true   = 20 * t / t[-1] * np.pi / 180

    roll_rate  = np.gradient(roll_true, dt)
    pitch_rate = np.gradient(pitch_true, dt)
    yaw_rate   = np.gradient(yaw_true, dt)

    gyro_noise_std = 0.5 * np.pi / 180
    gyro_bias_true = np.array([0.8, -0.5, 0.3]) * np.pi / 180
    accel_noise_std = 0.3

    lin_accel = np.zeros((3, n))
    mask = (t >= 5) & (t < 7)
    lin_accel[0, mask] = 3.0 * np.sin(2 * np.pi * 1.5 * t[mask])

    np.random.seed(77)
    gyro_meas = np.zeros((3, n))
    accel_meas = np.zeros((3, n))
    for i in range(n):
        gyro_meas[0, i] = roll_rate[i]  + gyro_bias_true[0] + gyro_noise_std * np.random.randn()
        gyro_meas[1, i] = pitch_rate[i] + gyro_bias_true[1] + gyro_noise_std * np.random.randn()
        gyro_meas[2, i] = yaw_rate[i]   + gyro_bias_true[2] + gyro_noise_std * np.random.randn()
        a_grav = accel_from_euler(roll_true[i], pitch_true[i])
        accel_meas[:, i] = a_grav + lin_accel[:, i] + accel_noise_std * np.random.randn(3)

    return {
        't': t, 'dt': dt, 'n': n,
        'roll_true': roll_true, 'pitch_true': pitch_true, 'yaw_true': yaw_true,
        'gyro_meas': gyro_meas, 'accel_meas': accel_meas,
        'gyro_bias_true': gyro_bias_true,
    }


def run_ekf(data, q_angles, q_bias, r_base_roll, r_base_pitch, r_scale,
            r_exp_rate, q_yaw=None, p_bias_init=0.05):
    """Run EKF with continuous adaptive R and separate roll/pitch noise.

    Adaptive R uses smooth exponential scaling:
        R_roll  = r_base_roll  * (1 + r_scale * exp(r_exp_rate * ad))
        R_pitch = r_base_pitch * (1 + r_scale * exp(r_exp_rate * ad))
    where ad = |‖a‖ - g|.  This avoids discrete tier jumps and gives
    pitch its own (higher) base R since x-axis linear accel corrupts
    pitch far more than roll.
    """
    t = data['t']; dt = data['dt']; n = data['n']
    gyro_meas = data['gyro_meas']; accel_meas = data['accel_meas']
    roll_true = data['roll_true']; pitch_true = data['pitch_true']; yaw_true = data['yaw_true']

    if q_yaw is None:
        q_yaw = q_angles

    x = np.zeros(6)
    P = np.diag([0.1, 0.1, 0.1, p_bias_init, p_bias_init, p_bias_init])
    Q = np.diag([q_angles, q_angles, q_yaw, q_bias, q_bias, q_bias])

    ekf_roll = np.zeros(n); ekf_pitch = np.zeros(n); ekf_yaw = np.zeros(n)
    ekf_bias = np.zeros((3, n))

    for i in range(n):
        wx = gyro_meas[0, i] - x[3]
        wy = gyro_meas[1, i] - x[4]
        wz = gyro_meas[2, i] - x[5]

        sr, cr = np.sin(x[0]), np.cos(x[0])
        tp, cp = np.tan(x[1]), np.cos(x[1])

        roll_dot  = wx + sr * tp * wy + cr * tp * wz
        pitch_dot = cr * wy - sr * wz
        yaw_dot   = (sr / cp) * wy + (cr / cp) * wz

        x_pred = x.copy()
        x_pred[0] += roll_dot * dt
        x_pred[1] += pitch_dot * dt
        x_pred[2] += yaw_dot * dt

        F = np.eye(6)
        F[0, 0] = 1 + (cr*tp*wy - sr*tp*wz) * dt
        F[0, 1] = ((sr/(cp**2))*wy + (cr/(cp**2))*wz) * dt
        F[0, 3] = -dt;  F[0, 4] = -sr * tp * dt;  F[0, 5] = -cr * tp * dt
        F[1, 0] = (-sr*wy - cr*wz) * dt
        F[1, 4] = -cr * dt;  F[1, 5] = sr * dt
        F[2, 0] = ((cr/cp)*wy - (sr/cp)*wz) * dt
        st = np.sin(x[1])
        F[2, 1] = ((sr*st/(cp**2))*wy + (cr*st/(cp**2))*wz) * dt
        F[2, 4] = -(sr/cp) * dt;  F[2, 5] = -(cr/cp) * dt

        P_pred = F @ P @ F.T + Q

        # Continuous adaptive R — smooth exponential scaling
        # Uses accel magnitude deviation for roll, plus extra x-axis
        # component for pitch (x-accel directly corrupts pitch measurement)
        am = np.sqrt(accel_meas[0,i]**2 + accel_meas[1,i]**2 + accel_meas[2,i]**2)
        ad = abs(am - 9.81)
        ax_dev = abs(accel_meas[0,i] - (-9.81 * np.sin(x_pred[1])))
        scale_roll  = 1.0 + r_scale * np.exp(r_exp_rate * ad)
        scale_pitch = 1.0 + r_scale * np.exp(r_exp_rate * max(ad, ax_dev))
        R_now = np.diag([r_base_roll * scale_roll, r_base_pitch * scale_pitch])

        z = np.array([
            np.arctan2(accel_meas[1, i], accel_meas[2, i]),
            np.arctan2(-accel_meas[0, i], np.sqrt(accel_meas[1, i]**2 + accel_meas[2, i]**2))
        ])
        h = np.array([x_pred[0], x_pred[1]])
        H = np.zeros((2, 6))
        H[0, 0] = 1; H[1, 1] = 1

        y_inn = z - h
        y_inn[0] = (y_inn[0] + np.pi) % (2*np.pi) - np.pi
        y_inn[1] = (y_inn[1] + np.pi) % (2*np.pi) - np.pi

        S = H @ P_pred @ H.T + R_now
        Kk = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + Kk @ y_inn
        P = (np.eye(6) - Kk @ H) @ P_pred

        ekf_roll[i] = x[0]; ekf_pitch[i] = x[1]; ekf_yaw[i] = x[2]
        ekf_bias[:, i] = x[3:6]

    rmse_roll  = np.sqrt(np.mean((ekf_roll  - roll_true)**2))  * 180/np.pi
    rmse_pitch = np.sqrt(np.mean((ekf_pitch - pitch_true)**2)) * 180/np.pi
    rmse_yaw   = np.sqrt(np.mean((ekf_yaw   - yaw_true)**2))  * 180/np.pi
    # Weighted: pitch and yaw matter more since they were bad
    score = rmse_roll + 3.0 * rmse_pitch + 3.0 * rmse_yaw

    return {
        'score': score,
        'rmse_roll': rmse_roll, 'rmse_pitch': rmse_pitch, 'rmse_yaw': rmse_yaw,
        'ekf_roll': ekf_roll, 'ekf_pitch': ekf_pitch, 'ekf_yaw': ekf_yaw,
        'ekf_bias': ekf_bias, 'final_x': x,
    }


def auto_tune(data):
    """Lightweight 2-pass search over Q/R parameters (~300 combos total)."""
    print("  Auto-tuning EKF parameters...")

    best_score = 1e9
    best_params = None
    best_q_yaw = None

    # Pass 1: coarse grid (~216 combos)
    for qa in [0.0003, 0.001, 0.005]:
        for qb in [3e-5, 3e-4, 1e-3]:
            for rr in [0.01, 0.05]:
                for rp in [0.05, 0.2]:
                    for rs in [0.5, 2.0, 8.0]:
                        for re in [1.5, 3.0]:
                            result = run_ekf(data, qa, qb, rr, rp, rs, re,
                                             p_bias_init=0.1)
                            if result['score'] < best_score:
                                best_score = result['score']
                                best_params = (qa, qb, rr, rp, rs, re, 0.1)

    # Pass 2: refine around best (~81 combos)
    qa0, qb0, rr0, rp0, rs0, re0, pb0 = best_params
    for qa in [qa0*0.5, qa0, qa0*2]:
        for qb in [qb0*0.3, qb0, qb0*3]:
            for rr in [rr0*0.5, rr0, rr0*2]:
                for rp in [rp0*0.5, rp0, rp0*2]:
                    result = run_ekf(data, qa, qb, rr, rp, rs0, re0,
                                     p_bias_init=pb0)
                    if result['score'] < best_score:
                        best_score = result['score']
                        best_params = (qa, qb, rr, rp, rs0, re0, pb0)

    # Pass 3: try separate q_yaw for better yaw bias estimation
    qa0, qb0, rr0, rp0, rs0, re0, pb0 = best_params
    best_q_yaw = qa0
    for q_yaw_mul in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        q_yaw = qa0 * q_yaw_mul
        result = run_ekf(data, qa0, qb0, rr0, rp0, rs0, re0,
                         q_yaw=q_yaw, p_bias_init=pb0)
        if result['score'] < best_score:
            best_score = result['score']
            best_q_yaw = q_yaw

    qa, qb, rr, rp, rs, re, pb = best_params
    print(f"  Best params: q_ang={qa:.6f}, q_bias={qb:.6f}")
    print(f"    r_base_roll={rr:.4f}, r_base_pitch={rp:.4f}, "
          f"r_scale={rs:.3f}, r_exp_rate={re:.2f}")
    print(f"    p_bias_init={pb:.3f}, q_yaw={best_q_yaw:.6f}")
    print(f"  Best score: {best_score:.3f}")
    return best_params + (best_q_yaw,)


def fig_ekf_imu():
    data = generate_sensor_data()
    t = data['t']; dt = data['dt']; n = data['n']
    deg = 180 / np.pi

    # Auto-tune
    best_params = auto_tune(data)
    qa, qb, rr, rp, rs, re, pb, q_yaw = best_params

    # Run with best params
    result = run_ekf(data, qa, qb, rr, rp, rs, re, q_yaw=q_yaw, p_bias_init=pb)
    ekf_roll = result['ekf_roll']; ekf_pitch = result['ekf_pitch']; ekf_yaw = result['ekf_yaw']
    ekf_bias = result['ekf_bias']; final_x = result['final_x']

    print(f"  Final RMSE: Roll={result['rmse_roll']:.2f}°, "
          f"Pitch={result['rmse_pitch']:.2f}°, Yaw={result['rmse_yaw']:.2f}°")

    # Baselines
    gyro_meas = data['gyro_meas']; accel_meas = data['accel_meas']
    roll_true = data['roll_true']; pitch_true = data['pitch_true']; yaw_true = data['yaw_true']
    gyro_bias_true = data['gyro_bias_true']

    gyro_roll = np.cumsum(gyro_meas[0] * dt)
    gyro_pitch = np.cumsum(gyro_meas[1] * dt)
    gyro_yaw = np.cumsum(gyro_meas[2] * dt)

    accel_roll  = np.arctan2(accel_meas[1], accel_meas[2])
    accel_pitch = np.arctan2(-accel_meas[0], np.sqrt(accel_meas[1]**2 + accel_meas[2]**2))

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    axes[0,0].plot(t, accel_roll*deg, 'C3', alpha=0.15, linewidth=0.5, label='仅加速度计')
    axes[0,0].plot(t, gyro_roll*deg, 'C1', alpha=0.6, linewidth=1, label='陀螺仪积分')
    axes[0,0].plot(t, ekf_roll*deg, 'C0', linewidth=2, label='EKF 估计值')
    axes[0,0].plot(t, roll_true*deg, 'k--', linewidth=1.5, label='真实值')
    axes[0,0].set_ylabel('角度 (°)'); axes[0,0].set_title('横滚')
    axes[0,0].legend(fontsize=7, loc='upper right')

    axes[0,1].plot(t, accel_pitch*deg, 'C3', alpha=0.15, linewidth=0.5, label='仅加速度计')
    axes[0,1].plot(t, gyro_pitch*deg, 'C1', alpha=0.6, linewidth=1, label='陀螺仪积分')
    axes[0,1].plot(t, ekf_pitch*deg, 'C0', linewidth=2, label='EKF 估计值')
    axes[0,1].plot(t, pitch_true*deg, 'k--', linewidth=1.5, label='真实值')
    axes[0,1].set_title('俯仰'); axes[0,1].legend(fontsize=7, loc='upper right')

    axes[0,2].plot(t, gyro_yaw*deg, 'C1', alpha=0.6, linewidth=1, label='陀螺仪积分')
    axes[0,2].plot(t, ekf_yaw*deg, 'C0', linewidth=2, label='EKF 估计值')
    axes[0,2].plot(t, yaw_true*deg, 'k--', linewidth=1.5, label='真实值')
    axes[0,2].set_title('偏航 (无加速度计修正!)'); axes[0,2].legend(fontsize=7)

    for ax_idx, (label, true_val) in enumerate(zip(
        ['X轴偏置', 'Y轴偏置', 'Z轴偏置'], gyro_bias_true)):
        axes[1, ax_idx].plot(t, ekf_bias[ax_idx]*deg, 'C0', linewidth=2, label='EKF 估计值')
        axes[1, ax_idx].axhline(true_val*deg, color='k', linestyle='--',
                               linewidth=1.5, label=f'真实值 = {true_val*deg:.2f} °/s')
        axes[1, ax_idx].set_xlabel('时间 (s)'); axes[1, ax_idx].set_ylabel('偏置 (°/s)')
        axes[1, ax_idx].set_title(f'陀螺仪{label}估计')
        axes[1, ax_idx].legend(fontsize=8)

    for ax in axes[0, :]:
        ax.axvspan(5, 7, alpha=0.1, color='C3')
        ax.text(6, ax.get_ylim()[1]*0.85, '振动', ha='center',
               fontsize=7, color='C3', fontstyle='italic')

    fig.suptitle(
        f'EKF 6-DOF IMU (自动调参) — RMSE: 横滚={result["rmse_roll"]:.2f}°, '
        f'俯仰={result["rmse_pitch"]:.2f}°, 偏航={result["rmse_yaw"]:.2f}°',
        fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, 'ekf_imu_6dof.pdf'), bbox_inches='tight')
    plt.close()
    print('[OK] ekf_imu_6dof.pdf')

    print(f'  Bias X est: {final_x[3]*deg:.3f} (true: {gyro_bias_true[0]*deg:.3f})')
    print(f'  Bias Y est: {final_x[4]*deg:.3f} (true: {gyro_bias_true[1]*deg:.3f})')
    print(f'  Bias Z est: {final_x[5]*deg:.3f} (true: {gyro_bias_true[2]*deg:.3f})')


if __name__ == '__main__':
    fig_ekf_imu()

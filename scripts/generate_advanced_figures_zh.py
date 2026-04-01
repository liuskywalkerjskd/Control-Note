#!/usr/bin/env python3
"""
Generate advanced PID improvement figures and LQG comparison (Chinese labels).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from zh_config import apply_zh
apply_zh()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import solve_continuous_are

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures_zh')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (8, 4.5),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'savefig.dpi': 200,
})

# ============================================================================
# Figure: Integral Separation — standard PID vs integral-separated PID
# ============================================================================
def fig_integral_separation():
    dt = 0.001
    t = np.arange(0, 3, dt)
    setpoint = np.where(t >= 0.2, 100.0, 0.0)

    # Plant: second-order motor
    J = 0.05; friction = 0.8

    def sim(integral_sep_threshold=None):
        kp, ki, kd = 8.0, 4.0, 0.8
        pos = 0; vel = 0; integral = 0; prev_pos = 0
        positions = []; outputs = []
        for i in range(len(t)):
            e = setpoint[i] - pos
            # Integral separation: only integrate when error is small
            if integral_sep_threshold is None:
                integral += ki * e * dt
            else:
                if abs(e) < integral_sep_threshold:
                    integral += ki * e * dt
                # else: do not accumulate integral
            integral = np.clip(integral, -5000, 5000)
            d = -(pos - prev_pos) / dt if i > 0 else 0
            out = kp * e + integral + kd * d
            out = np.clip(out, -30000, 30000)
            acc = (out - friction * vel) / J
            vel += acc * dt
            prev_pos = pos
            pos += vel * dt
            positions.append(pos)
            outputs.append(out)
        return np.array(positions), np.array(outputs)

    y_std, u_std = sim(None)
    y_sep, u_sep = sim(30.0)  # only integrate when |error| < 30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(t, setpoint, 'k--', linewidth=1, label='设定值')
    ax1.plot(t, y_std, 'C3', linewidth=2, label='标准 PID')
    ax1.plot(t, y_sep, 'C0', linewidth=2, label='积分分离 (ε=30)')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('位置')
    ax1.set_title('阶跃响应：标准 PID 与积分分离对比')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-10, 140)

    ax2.plot(t[:1000], u_std[:1000], 'C3', alpha=0.7, linewidth=1.5, label='标准 PID')
    ax2.plot(t[:1000], u_sep[:1000], 'C0', linewidth=2, label='积分分离')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('控制输出')
    ax2.set_title('控制信号 — 前一秒')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'integral_separation.pdf'))
    plt.close()
    print('[OK] integral_separation.pdf')


# ============================================================================
# Figure: PI-D vs I-PD vs PID — derivative placement comparison
# ============================================================================
def fig_derivative_placement():
    dt = 0.001
    t = np.arange(0, 3, dt)
    setpoint = np.where(t >= 0.1, 90.0, 0.0)

    J = 0.05; friction = 0.5
    kp, ki, kd = 10.0, 15.0, 0.6

    def sim(mode='PID'):
        pos = 0; vel = 0; integral = 0
        prev_e = 0; prev_pos = 0; prev_sp = 0
        positions = []; outputs = []
        for i in range(len(t)):
            e = setpoint[i] - pos
            integral += ki * e * dt
            integral = np.clip(integral, -5000, 5000)

            if mode == 'PID':
                # Standard: derivative on error
                d = kd * (e - prev_e) / dt if i > 0 else 0
                out = kp * e + integral + d
            elif mode == 'PI-D':
                # Derivative on measurement only
                d = -kd * (pos - prev_pos) / dt if i > 0 else 0
                out = kp * e + integral + d
            elif mode == 'I-PD':
                # P and D both on measurement
                d = -kd * (pos - prev_pos) / dt if i > 0 else 0
                out = -kp * pos + ki * np.sum(setpoint[:i+1] - np.array(positions + [pos])) * dt / max(i+1,1)
                # Simpler: I on error, P and D on measurement
                out = kp * (-pos) + integral + d + kp * setpoint[i]  # rearranged
                # Actually let's do it properly
                out = -kp * pos + integral + d
                # The integral drives to setpoint
                # Recalculate integral correctly: Ki * integral(r - y)
                pass

            out = np.clip(out, -30000, 30000)
            acc = (out - friction * vel) / J
            vel += acc * dt
            prev_e = e; prev_pos = pos; prev_sp = setpoint[i]
            pos += vel * dt
            positions.append(pos)
            outputs.append(out)
        return np.array(positions), np.array(outputs)

    # Let me redo this more carefully
    def sim2(d_on_error=True, p_on_error=True):
        pos = 0; vel = 0; integral = 0
        prev_e = 0; prev_y = 0
        positions = []; outputs = []; d_terms = []
        for i in range(len(t)):
            e = setpoint[i] - pos
            integral += ki * e * dt
            integral = np.clip(integral, -5000, 5000)
            p_term = kp * e if p_on_error else kp * (setpoint[i] - pos)
            if i == 0:
                d_term = 0
            elif d_on_error:
                d_term = kd * (e - prev_e) / dt
            else:
                d_term = -kd * (pos - prev_y) / dt
            out = p_term + integral + d_term
            out = np.clip(out, -30000, 30000)
            acc = (out - friction * vel) / J
            vel += acc * dt
            prev_e = e; prev_y = pos
            pos += vel * dt
            positions.append(pos)
            outputs.append(out)
            d_terms.append(d_term)
        return np.array(positions), np.array(outputs), np.array(d_terms)

    # I-PD: P on measurement, D on measurement
    def sim_ipd():
        pos = 0; vel = 0; integral = 0; prev_y = 0
        positions = []; outputs = []; d_terms = []
        for i in range(len(t)):
            e = setpoint[i] - pos
            integral += ki * e * dt
            integral = np.clip(integral, -10000, 10000)
            p_term = -kp * pos
            d_term = -kd * (pos - prev_y) / dt if i > 0 else 0
            out = p_term + integral + d_term
            out = np.clip(out, -30000, 30000)
            acc = (out - friction * vel) / J
            vel += acc * dt
            prev_y = pos
            pos += vel * dt
            positions.append(pos)
            outputs.append(out)
            d_terms.append(d_term)
        return np.array(positions), np.array(outputs), np.array(d_terms)

    y_pid, u_pid, d_pid = sim2(d_on_error=True, p_on_error=True)
    y_pid2, u_pid2, d_pid2 = sim2(d_on_error=False, p_on_error=True)  # PI-D
    y_ipd, u_ipd, d_ipd = sim_ipd()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(t, setpoint, 'k--', linewidth=1, label='设定值')
    ax1.plot(t, y_pid, 'C3', linewidth=2, label='PID (微分作用于误差) — 冲击！')
    ax1.plot(t, y_pid2, 'C0', linewidth=2, label='PI-D (微分作用于测量值)')
    ax1.plot(t, y_ipd, 'C2', linewidth=2, label='I-PD (比例、微分作用于测量值)')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('位置')
    ax1.set_title('阶跃响应：微分位置对比')
    ax1.legend(fontsize=8)
    ax1.set_ylim(-10, 130)

    # Right panel: show the DERIVATIVE TERM only — this is the "kick"
    n_show = 600  # first 0.6s
    ax2.plot(t[:n_show], d_pid[:n_show], 'C3', linewidth=2,
             label=f'PID: Kd·(de/dt) — 微分冲击尖峰 {d_pid[100:110].max():.0f}!')
    ax2.plot(t[:n_show], d_pid2[:n_show], 'C0', linewidth=2,
             label='PI-D: −Kd·(dy/dt) — 无冲击')
    ax2.plot(t[:n_show], d_ipd[:n_show], 'C2', linewidth=2,
             label='I-PD: −Kd·(dy/dt) — 无冲击')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('微分项输出')
    ax2.set_title('仅微分项 — 冲击的来源')
    ax2.legend(fontsize=7)
    # Clip y-axis to see PI-D and I-PD, annotate the spike
    d_max = max(abs(d_pid2[:n_show]).max(), abs(d_ipd[:n_show]).max())
    ax2.set_ylim(-d_max * 3, d_max * 6)
    kick_val = d_pid[100:110].max()
    ax2.annotate(f'微分冲击：{kick_val:.0f}',
                xy=(0.101, min(kick_val, d_max * 5.5)),
                fontsize=8, color='C3', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='C3', lw=1.5),
                xytext=(0.25, d_max * 4))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'derivative_placement.pdf'))
    plt.close()
    print('[OK] derivative_placement.pdf')


# ============================================================================
# Figure: PID Bode comparison — ideal PID vs filtered PID
# ============================================================================
def fig_pid_bode_comparison():
    kp = 10.0; ki = 2.0; kd = 1.0
    N = 10  # derivative filter coefficient

    # Ideal PID: C(s) = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki) / s
    num_ideal = [kd, kp, ki]
    den_ideal = [1, 0]

    # Filtered PID: C(s) = Kp + Ki/s + Kd*N*s/(s+N)
    # = [Kd*N*s^2 + (Kp*N + Kd*N*0)*s^2... let me compute properly
    # C(s) = Kp + Ki/s + Kd*s*N/(s+N)
    # Common denominator: s*(s+N)
    # = [Kp*s*(s+N) + Ki*(s+N) + Kd*N*s^2] / [s*(s+N)]
    # Numerator: (Kp+Kd*N)*s^2 + (Kp*N+Ki)*s + Ki*N
    num_filtered = [kp + kd*N, kp*N + ki, ki*N]
    den_filtered = [1, N, 0]

    w = np.logspace(-2, 4, 2000)

    # Compute frequency responses manually
    s = 1j * w
    H_ideal = (kd * s**2 + kp * s + ki) / s
    H_filtered = ((kp + kd*N) * s**2 + (kp*N + ki) * s + ki*N) / (s * (s + N))

    mag_ideal = 20 * np.log10(np.abs(H_ideal))
    mag_filtered = 20 * np.log10(np.abs(H_filtered))
    phase_ideal = np.degrees(np.angle(H_ideal))
    phase_filtered = np.degrees(np.angle(H_filtered))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax1.semilogx(w, mag_ideal, 'C3', linewidth=2, label='理想 PID (D 增益 → ∞)')
    ax1.semilogx(w, mag_filtered, 'C0', linewidth=2, label=f'滤波 PID (N={N})')
    ax1.axvline(N, color='C1', linestyle=':', alpha=0.6)
    ax1.annotate(f'D 滤波截止频率\nω = N = {N} rad/s', xy=(N, 25), fontsize=9, color='C1')
    ax1.set_ylabel('幅值 (dB)')
    ax1.set_title('波特幅值图：理想 PID 与滤波 PID 对比 — 为何必须对微分项滤波')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-10, 80)
    ax1.fill_between(w, 40, 80, where=w>50, alpha=0.1, color='C3')
    ax1.annotate('噪声放大\n区域', xy=(200, 60), fontsize=9, color='C3', ha='center')

    ax2.semilogx(w, phase_ideal, 'C3', linewidth=2, label='理想 PID')
    ax2.semilogx(w, phase_filtered, 'C0', linewidth=2, label='滤波 PID')
    ax2.set_xlabel('频率 (rad/s)')
    ax2.set_ylabel('相位 (度)')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'pid_bode_comparison.pdf'))
    plt.close()
    print('[OK] pid_bode_comparison.pdf')


# ============================================================================
# Figure: PID + output LPF — effect on noise rejection
# ============================================================================
def fig_pid_output_lpf():
    dt = 0.001
    t = np.arange(0, 3, dt)
    setpoint = np.where(t >= 0.2, 90.0, 0.0)

    J = 0.05; friction = 0.5
    kp, ki, kd = 10.0, 2.0, 1.5
    noise_amp = 0.5  # measurement noise

    np.random.seed(42)

    def sim(output_lpf_alpha=None):
        pos = 0; vel = 0; integral = 0; prev_y = 0; u_prev = 0
        positions = []; outputs = []
        for i in range(len(t)):
            meas = pos + noise_amp * np.random.randn()
            e = setpoint[i] - meas
            integral += ki * e * dt
            integral = np.clip(integral, -5000, 5000)
            d = -kd * (meas - prev_y) / dt if i > 0 else 0
            out = kp * e + integral + d

            # Apply output LPF
            if output_lpf_alpha is not None:
                out = output_lpf_alpha * out + (1 - output_lpf_alpha) * u_prev

            out = np.clip(out, -30000, 30000)
            acc = (out - friction * vel) / J
            vel += acc * dt
            prev_y = meas; u_prev = out
            pos += vel * dt
            positions.append(pos)
            outputs.append(out)
        return np.array(positions), np.array(outputs)

    y_raw, u_raw = sim(None)
    y_lpf, u_lpf = sim(0.35)  # moderate output LPF (was 0.1, too slow)

    # Also: filtered derivative only
    np.random.seed(42)
    def sim_filtered_d():
        pos = 0; vel = 0; integral = 0; prev_y = 0; d_filt = 0
        positions = []; outputs = []
        N = 15
        alpha_d = dt * N / (1 + dt * N)
        for i in range(len(t)):
            meas = pos + noise_amp * np.random.randn()
            e = setpoint[i] - meas
            integral += ki * e * dt
            integral = np.clip(integral, -5000, 5000)
            d_raw = -(meas - prev_y) / dt if i > 0 else 0
            d_filt = alpha_d * d_raw + (1 - alpha_d) * d_filt
            d = kd * d_filt
            out = kp * e + integral + d
            out = np.clip(out, -30000, 30000)
            acc = (out - friction * vel) / J
            vel += acc * dt
            prev_y = meas
            pos += vel * dt
            positions.append(pos)
            outputs.append(out)
        return np.array(positions), np.array(outputs)

    np.random.seed(42)
    y_dfilt, u_dfilt = sim_filtered_d()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, setpoint, 'k--', linewidth=1)
    ax1.plot(t, y_raw, 'C3', alpha=0.5, linewidth=1, label='无滤波 — 含噪声微分')
    ax1.plot(t, y_dfilt, 'C0', linewidth=2, label='微分滤波 (N=15)')
    ax1.plot(t, y_lpf, 'C2', linewidth=2, label='输出低通滤波 (α=0.35)')
    ax1.set_ylabel('位置')
    ax1.set_title('带噪声测量的 PID — 滤波策略对比')
    ax1.legend(fontsize=9)

    ax2.plot(t[:2000], u_raw[:2000], 'C3', alpha=0.3, linewidth=0.5, label='未滤波 — 执行器损耗严重')
    ax2.plot(t[:2000], u_dfilt[:2000], 'C0', linewidth=1.5, label='微分滤波')
    ax2.plot(t[:2000], u_lpf[:2000], 'C2', linewidth=1.5, label='输出低通滤波')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('控制信号')
    ax2.set_title('控制输出 — 未滤波微分会损坏电机')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'pid_noise_filtering.pdf'))
    plt.close()
    print('[OK] pid_noise_filtering.pdf')


# ============================================================================
# Figure: LQG vs PID — noisy balance bot comparison
# ============================================================================
def fig_lqg_comparison():
    # Stable-ish inverted pendulum — long pendulum, heavy cart
    g = 9.81; L = 1.0; M = 3.0

    A = np.array([
        [0, 1, 0, 0],
        [g/L, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    B = np.array([[0], [-1/(M*L)], [0], [1/M]])
    C = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

    # LQR — well-tuned
    Q_lqr = np.diag([200, 5, 15, 3])
    R_lqr = np.array([[1.0]])
    P_lqr = solve_continuous_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(R_lqr) @ B.T @ P_lqr

    # VERY HIGH measurement noise — Kalman's advantage is dramatic here
    # Angle noise std ≈ 8°, position noise std ≈ 0.2 m
    R_kf = np.diag([0.02, 0.04])
    Q_kf = np.diag([0.001, 0.1, 0.001, 0.05])

    dt = 0.002
    Ad = np.eye(4) + A * dt
    Bd = B * dt

    # Solve steady-state Kalman gain
    P_est = np.eye(4) * 0.1
    for _ in range(5000):
        P_pred = Ad @ P_est @ Ad.T + Q_kf
        S = C @ P_pred @ C.T + R_kf
        Kk = P_pred @ C.T @ np.linalg.inv(S)
        P_est = (np.eye(4) - Kk @ C) @ P_pred

    t = np.arange(0, 6, dt)
    np.random.seed(42)
    meas_noise_angle = np.sqrt(R_kf[0,0]) * np.random.randn(len(t))
    meas_noise_pos = np.sqrt(R_kf[1,1]) * np.random.randn(len(t))

    # Disturbance: impulse at t=3s
    disturbance = np.zeros(len(t))
    disturbance[int(3.0/dt):int(3.03/dt)] = 10.0

    # ---- LQG (LQR + Kalman) ----
    x_true = np.array([[0.06], [0], [0], [0]])  # ~3.4 deg
    x_hat = np.zeros((4, 1))
    P_kf_run = np.eye(4) * 0.1
    lqg_states = np.zeros((4, len(t)))
    lqg_ctrl = np.zeros(len(t))

    for i in range(len(t)):
        lqg_states[:, i] = x_true.flatten()
        u_val = np.clip(float((-K @ x_hat)[0, 0]), -40, 40)
        lqg_ctrl[i] = u_val

        x_true = x_true + (A @ x_true + B * (u_val + disturbance[i])) * dt

        x_hat_pred = Ad @ x_hat + Bd * u_val
        P_pred = Ad @ P_kf_run @ Ad.T + Q_kf
        y = C @ x_true + np.array([[meas_noise_angle[i]], [meas_noise_pos[i]]])
        S = C @ P_pred @ C.T + R_kf
        Kk = P_pred @ C.T @ np.linalg.inv(S)
        x_hat = x_hat_pred + Kk @ (y - C @ x_hat_pred)
        P_kf_run = (np.eye(4) - Kk @ C) @ P_pred

    # ---- Noisy state feedback (same K, no Kalman) ----
    np.random.seed(42)
    x_pid = np.array([[0.06], [0], [0], [0]])
    pid_states = np.zeros((4, len(t)))
    pid_ctrl = np.zeros(len(t))
    prev_ma = 0.0; prev_mp = 0.0; lpf_r = 0.0; lpf_v = 0.0
    N_filt = 15; alpha_f = dt * N_filt / (1 + dt * N_filt)

    for i in range(len(t)):
        pid_states[:, i] = x_pid.flatten()
        ma = x_pid[0, 0] + meas_noise_angle[i]
        mp = x_pid[2, 0] + meas_noise_pos[i]
        rr = (ma - prev_ma) / dt if i > 0 else 0
        lpf_r = alpha_f * rr + (1 - alpha_f) * lpf_r
        vr = (mp - prev_mp) / dt if i > 0 else 0
        lpf_v = alpha_f * vr + (1 - alpha_f) * lpf_v

        u_pid = np.clip(float((-K @ np.array([[ma],[lpf_r],[mp],[lpf_v]]))[0, 0]), -40, 40)
        pid_ctrl[i] = u_pid
        prev_ma = ma; prev_mp = mp
        x_pid = x_pid + (A @ x_pid + B * (u_pid + disturbance[i])) * dt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0,0].plot(t, np.degrees(lqg_states[0]), 'C0', linewidth=2, label='LQG')
    axes[0,0].plot(t, np.degrees(pid_states[0]), 'C3', linewidth=1.5, alpha=0.8, label='带噪声状态反馈')
    axes[0,0].axhline(0, color='k', linestyle='--', alpha=0.2)
    axes[0,0].axvspan(3.0, 3.03, alpha=0.3, color='C1', label='扰动')
    axes[0,0].set_ylabel('倾斜角 (°)')
    axes[0,0].set_title('倾斜角')
    axes[0,0].legend(fontsize=8)

    axes[0,1].plot(t, lqg_states[2]*100, 'C0', linewidth=2, label='LQG')
    axes[0,1].plot(t, pid_states[2]*100, 'C3', linewidth=1.5, alpha=0.8, label='带噪声状态反馈')
    axes[0,1].axvspan(3.0, 3.03, alpha=0.3, color='C1')
    axes[0,1].set_ylabel('位置 (cm)')
    axes[0,1].set_title('轮子位置')
    axes[0,1].legend(fontsize=8)

    axes[1,0].plot(t, lqg_ctrl, 'C0', linewidth=1.5, label='LQG — 平滑')
    axes[1,0].plot(t, pid_ctrl, 'C3', linewidth=0.5, alpha=0.4, label='带噪声反馈 — 抖动')
    axes[1,0].axvspan(3.0, 3.03, alpha=0.3, color='C1')
    axes[1,0].set_xlabel('时间 (s)')
    axes[1,0].set_ylabel('力 (N)')
    axes[1,0].set_title('控制力')
    axes[1,0].legend(fontsize=8)

    # Bar chart
    t_start = 500
    rms_lqg = np.degrees(np.sqrt(np.mean(lqg_states[0, t_start:]**2)))
    rms_pid = np.degrees(np.sqrt(np.mean(pid_states[0, t_start:]**2)))
    rms_cu_lqg = np.sqrt(np.mean(lqg_ctrl[t_start:]**2))
    rms_cu_pid = np.sqrt(np.mean(pid_ctrl[t_start:]**2))
    x_pos = np.array([0, 1])
    w = 0.35
    bars1 = axes[1,1].bar(x_pos - w/2, [rms_lqg, rms_pid], w, color=['C0', 'C3'])
    ax_r = axes[1,1].twinx()
    bars2 = ax_r.bar(x_pos + w/2, [rms_cu_lqg, rms_cu_pid], w,
                     color=['C0', 'C3'], alpha=0.3)
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(['LQG', '带噪声反馈'])
    axes[1,1].set_ylabel('RMS 倾斜误差 (°)')
    ax_r.set_ylabel('RMS 控制力 (N)')
    axes[1,1].set_title('性能对比（越低越好）')
    for bar, val in zip(bars1, [rms_lqg, rms_pid]):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{val:.2f}°', ha='center', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, [rms_cu_lqg, rms_cu_pid]):
        ax_r.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}N', ha='center', fontsize=9, alpha=0.6)

    fig.suptitle('LQG 对比 带噪声状态反馈 — 平衡小车 (t=3s 时施加扰动)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'lqg_comparison.pdf'))
    plt.close()
    print('[OK] lqg_comparison.pdf')


# ============================================================================
# Figure: Two-DOF PID — setpoint weighting effect
# ============================================================================
def fig_two_dof_pid():
    dt = 0.001
    t = np.arange(0, 2, dt)

    # Two setpoint changes
    setpoint = np.where(t >= 0.1, 90.0, 0.0)
    # Add a disturbance
    disturbance = np.where((t >= 1.0) & (t < 1.05), -3000.0, 0.0)

    J = 0.05; friction = 0.5
    kp, ki, kd = 10.0, 2.0, 1.5

    def sim_2dof(b_weight, c_weight):
        """b: setpoint weight for P, c: setpoint weight for D"""
        pos = 0; vel = 0; integral = 0; prev_y = 0; prev_r = 0
        positions = []
        for i in range(len(t)):
            r = setpoint[i]
            e = r - pos
            integral += ki * e * dt
            integral = np.clip(integral, -5000, 5000)

            p_term = kp * (b_weight * r - pos)
            d_term = kd * ((c_weight * (r - prev_r) - (pos - prev_y)) / dt) if i > 0 else 0

            out = p_term + integral + d_term + disturbance[i]
            out = np.clip(out, -30000, 30000)
            acc = (out - friction * vel) / J
            vel += acc * dt
            prev_y = pos; prev_r = r
            pos += vel * dt
            positions.append(pos)
        return np.array(positions)

    y_11 = sim_2dof(1.0, 1.0)  # Standard PID (b=1, c=1)
    y_10 = sim_2dof(1.0, 0.0)  # PI-D (b=1, c=0)
    y_07 = sim_2dof(0.7, 0.0)  # 2-DOF (b=0.7, c=0)
    y_05 = sim_2dof(0.5, 0.0)  # 2-DOF (b=0.5, c=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Zoom on step response
    ax1.plot(t[:800], setpoint[:800], 'k--', linewidth=1)
    ax1.plot(t[:800], y_11[:800], 'C3', linewidth=2, label='b=1, c=1 (标准 PID)')
    ax1.plot(t[:800], y_10[:800], 'C1', linewidth=2, label='b=1, c=0 (PI-D)')
    ax1.plot(t[:800], y_07[:800], 'C0', linewidth=2, label='b=0.7, c=0 (二自由度 PID)')
    ax1.plot(t[:800], y_05[:800], 'C2', linewidth=2, label='b=0.5, c=0 (二自由度 PID)')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('位置')
    ax1.set_title('设定值跟踪 — b 越小超调越小')
    ax1.legend(fontsize=8)

    # Zoom on disturbance rejection
    ax2.plot(t[900:1800], setpoint[900:1800], 'k--', linewidth=1)
    ax2.plot(t[900:1800], y_11[900:1800], 'C3', linewidth=2, label='b=1 (标准)')
    ax2.plot(t[900:1800], y_10[900:1800], 'C1', linewidth=2, label='b=1 (PI-D)')
    ax2.plot(t[900:1800], y_07[900:1800], 'C0', linewidth=2, label='b=0.7')
    ax2.plot(t[900:1800], y_05[900:1800], 'C2', linewidth=2, label='b=0.5')
    ax2.set_xlabel('时间 (s)')
    ax2.set_title('扰动抑制 — 所有曲线完全相同（b 不影响扰动抑制！）')
    ax2.legend(fontsize=8)

    fig.suptitle('二自由度 PID — 独立调节设定值跟踪与扰动抑制',
                fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'two_dof_pid.pdf'))
    plt.close()
    print('[OK] two_dof_pid.pdf')


# ============================================================================
if __name__ == '__main__':
    print('正在生成高级图形...')
    fig_integral_separation()
    fig_derivative_placement()
    fig_pid_bode_comparison()
    fig_pid_output_lpf()
    fig_lqg_comparison()
    fig_two_dof_pid()
    print(f'\n所有高级图形已保存到 {OUT}/')

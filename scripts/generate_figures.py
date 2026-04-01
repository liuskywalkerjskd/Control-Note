#!/usr/bin/env python3
"""
Generate all figures for "A Practical Guide to Control Theory"
Each figure demonstrates a key concept with visual clarity.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import solve_continuous_are
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')
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
# Figure 1: Frequency Domain — Clean signal + noise, FFT reveals structure
# ============================================================================
def fig_frequency_spectrum():
    fs = 1000
    t = np.arange(0, 1, 1/fs)
    # Signal: 5 Hz sine + 42 Hz resonance + noise
    clean = np.sin(2*np.pi*5*t)
    resonance = 0.3 * np.sin(2*np.pi*42*t)
    noise = 0.5 * np.random.randn(len(t))
    raw = clean + resonance + noise

    freqs = np.fft.rfftfreq(len(t), 1/fs)
    spectrum = np.abs(np.fft.rfft(raw)) / len(t) * 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5))

    ax1.plot(t[:300], raw[:300], 'C3', alpha=0.7, label='Raw signal (noisy)')
    ax1.plot(t[:300], clean[:300], 'C0', linewidth=2.5, label='True signal (5 Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time Domain — The signal looks like a mess')
    ax1.legend(loc='upper right')

    ax2.plot(freqs, spectrum, 'C0')
    ax2.axvline(5, color='C2', linestyle='--', alpha=0.8, label='Signal @ 5 Hz')
    ax2.axvline(42, color='C1', linestyle='--', alpha=0.8, label='Resonance @ 42 Hz')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Domain (FFT) — Now we can see everything clearly')
    ax2.set_xlim(0, 100)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'frequency_spectrum.pdf'))
    plt.close()
    print('[OK] frequency_spectrum.pdf')


# ============================================================================
# Figure 2: LPF — Step response with different cutoff frequencies
# ============================================================================
def fig_lpf_response():
    fs = 1000
    t = np.arange(0, 0.5, 1/fs)
    step = np.ones_like(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: step response for different cutoffs
    for fc in [5, 15, 50, 200]:
        wc = 2 * np.pi * fc
        sys_lpf = signal.lti([wc], [1, wc])
        tout, y, _ = signal.lsim(sys_lpf, step, t)
        ax1.plot(t*1000, y, label=f'$f_c$ = {fc} Hz')
    ax1.plot(t*1000, step, 'k--', alpha=0.3, label='Step input')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Output')
    ax1.set_title('LPF Step Response — Lower cutoff = slower rise')
    ax1.legend(fontsize=9)

    # Right: noise filtering demo
    t2 = np.arange(0, 1, 1/fs)
    clean = np.sin(2*np.pi*3*t2)
    noisy = clean + 0.5*np.random.randn(len(t2))
    alpha = 0.05  # ~16 Hz at 1kHz
    filtered = np.zeros_like(noisy)
    filtered[0] = noisy[0]
    for i in range(1, len(noisy)):
        filtered[i] = alpha * noisy[i] + (1-alpha) * filtered[i-1]

    ax2.plot(t2, noisy, 'C3', alpha=0.3, linewidth=0.5, label='Noisy')
    ax2.plot(t2, filtered, 'C0', linewidth=2, label='LPF filtered')
    ax2.plot(t2, clean, 'C2', linewidth=1.5, linestyle='--', label='True signal')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('LPF Noise Removal — Smoothing a 3 Hz signal')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'lpf_response.pdf'))
    plt.close()
    print('[OK] lpf_response.pdf')


# ============================================================================
# Figure 3: LPF Soft Start — Step command → smooth ramp
# ============================================================================
def fig_lpf_soft_start():
    fs = 1000
    t = np.arange(0, 2, 1/fs)
    setpoint = np.where(t >= 0.2, 5000, 0).astype(float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, setpoint, 'k--', linewidth=1.5, label='Raw step command')

    for tau_ms, color in [(50, 'C0'), (200, 'C1'), (500, 'C2')]:
        fc = 1.0 / (2*np.pi*(tau_ms/1000))
        alpha = (1/fs) / ((1/fs) + 1/(2*np.pi*fc))
        y = np.zeros_like(setpoint)
        for i in range(1, len(y)):
            y[i] = alpha * setpoint[i] + (1-alpha)*y[i-1]
        ax.plot(t, y, color=color, label=f'LPF soft start (τ = {tau_ms} ms)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor speed (RPM)')
    ax.set_title('LPF as Soft Start — One line of code replaces complex ramp generators')
    ax.legend()
    ax.set_ylim(-200, 5800)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'lpf_soft_start.pdf'))
    plt.close()
    print('[OK] lpf_soft_start.pdf')


# ============================================================================
# Figure 4: Second-order system — Step response for different damping ratios
# ============================================================================
def fig_second_order():
    wn = 10  # natural frequency
    t = np.arange(0, 2, 0.001)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(1, color='k', linestyle='--', alpha=0.3)

    zetas = [0.1, 0.3, 0.5, 0.707, 1.0, 2.0]
    colors = ['C3', 'C1', 'C4', 'C0', 'C2', 'C5']
    for zeta, c in zip(zetas, colors):
        sys2 = signal.lti([wn**2], [1, 2*zeta*wn, wn**2])
        tout, y = signal.step(sys2, T=t)
        label = f'ζ = {zeta}'
        if zeta == 0.707:
            label += ' (optimal)'
        ax.plot(t, y, color=c, label=label)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.set_title(f'Second-Order Step Response (ωn = {wn} rad/s) — Feel the damping ratio')
    ax.legend(loc='right', fontsize=9)
    ax.set_ylim(-0.1, 2.0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'second_order_response.pdf'))
    plt.close()
    print('[OK] second_order_response.pdf')


# ============================================================================
# Figure 5: PID tuning progression — P → PI → PID
# ============================================================================
def fig_pid_tuning():
    # Plant: motor G(s) = 100 / (s^2 + 10s + 20)  (underdamped)
    num_plant = [100]
    den_plant = [1, 10, 20]

    t = np.arange(0, 2, 0.001)

    configs = [
        ('P only: Kp=5', [5], [1]),
        ('PI: Kp=5, Ki=3', [5, 3], [1, 0]),
        ('PID: Kp=5, Ki=3, Kd=2', [10, 5, 3], [1, 0]),  # Kd*s^2 + Kp*s + Ki
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, (label, num_c, den_c) in zip(axes, configs):
        # Closed loop: C*G / (1 + C*G)
        num_ol = np.polymul(num_c, num_plant)
        den_ol = np.polymul(den_c, den_plant)
        num_cl = num_ol
        den_cl = np.polyadd(den_ol, num_ol)
        sys_cl = signal.lti(num_cl, den_cl)
        tout, y = signal.step(sys_cl, T=t)
        ax.plot(t, y, 'C0', linewidth=2)
        ax.axhline(1, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_title(label, fontsize=11)
        ax.set_ylim(-0.1, 1.8)
        # Mark overshoot
        mp = (np.max(y) - 1) * 100
        if mp > 1:
            ax.annotate(f'Overshoot: {mp:.0f}%',
                       xy=(t[np.argmax(y)], np.max(y)),
                       xytext=(0.8, 1.5), fontsize=9,
                       arrowprops=dict(arrowstyle='->', color='C3'),
                       color='C3')
        # Steady state error
        ss = abs(1 - y[-1])
        if ss > 0.01:
            ax.annotate(f'SS error: {ss:.2f}',
                       xy=(t[-1], y[-1]), xytext=(1.0, 0.4),
                       fontsize=9, color='C1',
                       arrowprops=dict(arrowstyle='->', color='C1'))

    axes[0].set_ylabel('Output')
    fig.suptitle('PID Tuning Progression on a Second-Order Plant', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'pid_tuning.pdf'))
    plt.close()
    print('[OK] pid_tuning.pdf')


# ============================================================================
# Figure 6: Bode Plot — Example with gain/phase margins marked
# ============================================================================
def fig_bode_plot():
    # Open-loop: C(s)*G(s) = 100 / (s*(s+2)*(s+10))
    num = [100]
    den = np.polymul(np.polymul([1, 0], [1, 2]), [1, 10])
    sys_ol = signal.lti(num, den)

    w = np.logspace(-2, 3, 1000)
    w, mag, phase = signal.bode(sys_ol, w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.semilogx(w, mag, 'C0', linewidth=2)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    # Gain crossover
    idx_gc = np.argmin(np.abs(mag))
    ax1.plot(w[idx_gc], mag[idx_gc], 'ro', markersize=8)
    ax1.annotate(f'Gain crossover\nω = {w[idx_gc]:.1f} rad/s',
                xy=(w[idx_gc], mag[idx_gc]), xytext=(w[idx_gc]*3, 15),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='C3'), color='C3')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Bode Plot — Reading Stability Margins')

    ax2.semilogx(w, phase, 'C0', linewidth=2)
    ax2.axhline(-180, color='k', linestyle='--', alpha=0.5)
    # Phase at gain crossover = phase margin
    pm = 180 + phase[idx_gc]
    ax2.annotate(f'Phase margin = {pm:.0f}°',
                xy=(w[idx_gc], phase[idx_gc]),
                xytext=(w[idx_gc]*3, -130),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='C2', linewidth=2),
                color='C2')
    ax2.plot(w[idx_gc], phase[idx_gc], 'ro', markersize=8)
    # Draw PM arrow
    ax2.annotate('', xy=(w[idx_gc], -180), xytext=(w[idx_gc], phase[idx_gc]),
                arrowprops=dict(arrowstyle='<->', color='C2', linewidth=1.5))

    # Gain margin
    idx_pc = np.argmin(np.abs(phase + 180))
    gm = -mag[idx_pc]
    ax1.annotate(f'Gain margin = {gm:.0f} dB',
                xy=(w[idx_pc], mag[idx_pc]),
                xytext=(w[idx_pc]/5, mag[idx_pc]+15),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='C1', linewidth=2),
                color='C1')
    ax1.plot(w[idx_pc], mag[idx_pc], 'rs', markersize=8)
    ax1.annotate('', xy=(w[idx_pc], 0), xytext=(w[idx_pc], mag[idx_pc]),
                arrowprops=dict(arrowstyle='<->', color='C1', linewidth=1.5))

    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel('Phase (degrees)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'bode_plot.pdf'))
    plt.close()
    print('[OK] bode_plot.pdf')


# ============================================================================
# Figure 7: Kalman Filter — Noisy measurement vs. estimate
# ============================================================================
def fig_kalman_filter():
    dt = 0.01
    t = np.arange(0, 10, dt)
    n = len(t)

    # True state: angle following a sine wave
    theta_true = 30 * np.sin(0.5 * t)  # degrees
    omega_true = 30 * 0.5 * np.cos(0.5 * t)  # deg/s

    # Simulated sensors
    gyro_bias = 1.5  # deg/s
    gyro_meas = omega_true + gyro_bias + 2.0*np.random.randn(n)
    accel_meas = theta_true + 5.0*np.random.randn(n)

    # Pure gyro integration (drifts)
    gyro_integrated = np.cumsum(gyro_meas * dt)

    # Complementary filter
    alpha_cf = 0.98
    cf_angle = np.zeros(n)
    cf_angle[0] = accel_meas[0]
    for i in range(1, n):
        cf_angle[i] = alpha_cf*(cf_angle[i-1] + gyro_meas[i]*dt) + (1-alpha_cf)*accel_meas[i]

    # Kalman filter: state = [theta, gyro_bias]
    A = np.array([[1, -dt], [0, 1]])
    B = np.array([[dt], [0]])
    C = np.array([[1, 0]])
    Q = np.diag([0.001, 0.003])
    R = np.array([[25.0]])  # accel noise variance

    x = np.array([[accel_meas[0]], [0.0]])
    P = np.eye(2)
    kf_angle = np.zeros(n)
    kf_bias = np.zeros(n)

    for i in range(n):
        # Predict
        u = np.array([[gyro_meas[i]]])
        x = A @ x + B @ u
        P = A @ P @ A.T + Q
        # Update
        y = accel_meas[i] - (C @ x)[0, 0]
        S = (C @ P @ C.T + R)[0, 0]
        K = (P @ C.T) / S
        x = x + K * y
        P = (np.eye(2) - K @ C) @ P

        kf_angle[i] = x[0, 0]
        kf_bias[i] = x[1, 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))

    ax1.plot(t, accel_meas, 'C3', alpha=0.15, linewidth=0.5, label='Accelerometer (noisy)')
    ax1.plot(t, gyro_integrated, 'C1', alpha=0.6, linewidth=1, label='Gyro integration (drifts)')
    ax1.plot(t, cf_angle, 'C4', linewidth=1.5, label='Complementary filter')
    ax1.plot(t, kf_angle, 'C0', linewidth=2, label='Kalman filter')
    ax1.plot(t, theta_true, 'k--', linewidth=1.5, label='True angle')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('IMU Tilt Estimation — Kalman Filter vs. Alternatives')
    ax1.legend(loc='upper right', fontsize=8)

    ax2.plot(t, kf_bias, 'C0', linewidth=2, label='Kalman estimated bias')
    ax2.axhline(gyro_bias, color='k', linestyle='--', label=f'True bias = {gyro_bias} °/s')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Gyro bias (°/s)')
    ax2.set_title('Kalman Filter Automatically Estimates Gyro Bias')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'kalman_filter.pdf'))
    plt.close()
    print('[OK] kalman_filter.pdf')


# ============================================================================
# Figure 8: LQR Balance Bot — State trajectories
# ============================================================================
def fig_lqr_balance():
    # Linearized inverted pendulum on wheels
    # States: [theta, theta_dot, x, x_dot]
    g = 9.81; L = 0.3; m = 1.0; M = 2.0
    a = g * (m + M) / (M * L)
    b_val = 1.0 / (M * L)

    A = np.array([
        [0, 1, 0, 0],
        [a, 0, 0, 0],
        [0, 0, 0, 1],
        [-m*g/M, 0, 0, 0]
    ])
    B = np.array([[0], [-b_val], [0], [1/M]])

    Q = np.diag([100, 1, 10, 1])
    R = np.array([[1.0]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    dt = 0.001
    t = np.arange(0, 3, dt)
    x = np.array([[0.15], [0], [0], [0]])  # initial tilt of ~8.6 degrees
    states = np.zeros((4, len(t)))
    controls = np.zeros(len(t))

    for i in range(len(t)):
        states[:, i] = x.flatten()
        u = -K @ x
        u_clamp = np.clip(u, -50, 50)
        controls[i] = u_clamp[0, 0]
        x_dot = A @ x + B @ u_clamp
        x = x + x_dot * dt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    axes[0,0].plot(t, np.degrees(states[0]), 'C0', linewidth=2)
    axes[0,0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0,0].set_ylabel('Tilt angle (°)')
    axes[0,0].set_title('θ — Tilt recovers from 8.6° push')

    axes[0,1].plot(t, np.degrees(states[1]), 'C1', linewidth=2)
    axes[0,1].set_ylabel('Angular rate (°/s)')
    axes[0,1].set_title('θ̇ — Angular rate')

    axes[1,0].plot(t, states[2]*100, 'C2', linewidth=2)
    axes[1,0].set_ylabel('Position (cm)')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_title('x — Wheel position')

    axes[1,1].plot(t, controls, 'C3', linewidth=2)
    axes[1,1].set_ylabel('Force (N)')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_title('u — Control effort')

    fig.suptitle('LQR Balance Bot — Q = diag(100,1,10,1), R = 1', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'lqr_balance.pdf'))
    plt.close()
    print('[OK] lqr_balance.pdf')


# ============================================================================
# Figure 9: Anti-windup — With and without integral clamping
# ============================================================================
def fig_anti_windup():
    dt = 0.001
    t = np.arange(0, 4, dt)
    setpoint = np.where(t < 1, 0, np.where(t < 2.5, 100, 0)).astype(float)

    kp, ki, kd = 2.0, 5.0, 0.1
    out_max = 50.0

    def sim_pid(anti_windup):
        # Simple plant: first-order with saturation
        y = 0.0; integral = 0.0; prev_e = 0.0; prev_y = 0.0
        ys = []; integrals = []; outputs = []
        for i in range(len(t)):
            e = setpoint[i] - y
            integral += ki * e * dt
            if anti_windup:
                integral = np.clip(integral, -out_max, out_max)
            d = -(y - prev_y) / dt
            out_raw = kp * e + integral + kd * d
            out = np.clip(out_raw, -out_max, out_max)
            # Plant dynamics
            y += (out - y * 0.5) * dt * 5
            prev_e = e; prev_y = y
            ys.append(y); integrals.append(integral); outputs.append(out)
        return np.array(ys), np.array(integrals), np.array(outputs)

    y_no, int_no, _ = sim_pid(False)
    y_aw, int_aw, _ = sim_pid(True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)

    ax1.plot(t, setpoint, 'k--', linewidth=1, label='Setpoint')
    ax1.plot(t, y_no, 'C3', linewidth=2, label='Without anti-windup')
    ax1.plot(t, y_aw, 'C0', linewidth=2, label='With anti-windup')
    ax1.set_ylabel('Output')
    ax1.set_title('Anti-Windup Effect — Saturation causes massive overshoot without clamping')
    ax1.legend()

    ax2.plot(t, int_no, 'C3', linewidth=2, label='Integral (no clamp) — explodes!')
    ax2.plot(t, int_aw, 'C0', linewidth=2, label='Integral (clamped)')
    ax2.axhline(out_max, color='k', linestyle=':', alpha=0.5, label='Clamp limit')
    ax2.axhline(-out_max, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Integral term')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'anti_windup.pdf'))
    plt.close()
    print('[OK] anti_windup.pdf')


# ============================================================================
# Figure 10: Cascaded PID vs Single PID — Disturbance rejection
# ============================================================================
def fig_cascaded_pid():
    dt = 0.001
    t = np.arange(0, 3, dt)
    setpoint = np.where(t >= 0.2, 90.0, 0.0)  # 90 degree step
    disturbance = np.where((t >= 1.5) & (t < 1.6), -500.0, 0.0)  # bump at t=1.5s

    # Simple motor model: J*a = torque - friction*omega
    J = 0.05; friction = 0.5

    def sim_single_pid():
        kp, ki, kd = 15.0, 2.0, 1.5
        pos = 0; vel = 0; integral = 0; prev_pos = 0
        positions = []
        for i in range(len(t)):
            e = setpoint[i] - pos
            integral += ki * e * dt
            integral = np.clip(integral, -5000, 5000)
            d = -(pos - prev_pos) / dt
            torque = kp * e + integral + kd * d
            torque = np.clip(torque, -30000, 30000) + disturbance[i]
            acc = (torque - friction * vel) / J
            vel += acc * dt
            prev_pos = pos
            pos += vel * dt
            positions.append(pos)
        return np.array(positions)

    def sim_cascaded():
        # Outer: position PID
        kp_o = 10.0; ki_o = 0.0; kd_o = 0.0
        # Inner: velocity PID
        kp_i = 80.0; ki_i = 5.0; kd_i = 0.0

        pos = 0; vel = 0
        int_o = 0; int_i = 0; prev_pos = 0; prev_vel = 0
        positions = []
        for i in range(len(t)):
            # Outer loop
            e_o = setpoint[i] - pos
            vel_cmd = kp_o * e_o
            vel_cmd = np.clip(vel_cmd, -500, 500)

            # Inner loop
            e_i = vel_cmd - vel
            int_i += ki_i * e_i * dt
            int_i = np.clip(int_i, -5000, 5000)
            torque = kp_i * e_i + int_i
            torque = np.clip(torque, -30000, 30000) + disturbance[i]

            acc = (torque - friction * vel) / J
            prev_vel = vel
            vel += acc * dt
            prev_pos = pos
            pos += vel * dt
            positions.append(pos)
        return np.array(positions)

    y_single = sim_single_pid()
    y_cascade = sim_cascaded()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, setpoint, 'k--', linewidth=1, label='Setpoint (90°)')
    ax.plot(t, y_single, 'C3', linewidth=2, label='Single PID')
    ax.plot(t, y_cascade, 'C0', linewidth=2, label='Cascaded PID')
    ax.axvline(1.5, color='C1', linestyle=':', alpha=0.6)
    ax.annotate('Disturbance\n(bump)', xy=(1.5, 60), fontsize=10, color='C1',
               ha='center')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Cascaded PID vs Single PID — Disturbance rejection comparison')
    ax.legend()
    ax.set_ylim(-10, 130)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'cascaded_pid.pdf'))
    plt.close()
    print('[OK] cascaded_pid.pdf')


# ============================================================================
# Figure 11: PID Architectures Comparison — Single vs Cascaded vs Parallel-Cascaded
# ============================================================================
def fig_pid_architectures():
    dt = 0.001
    t = np.arange(0, 4, dt)
    # Yaw: 90 deg step at t=0.2s; Pitch: 45 deg step at t=0.5s
    sp_yaw = np.where(t >= 0.2, 90.0, 0.0)
    sp_pitch = np.where(t >= 0.5, 45.0, 0.0)
    dist = np.where((t >= 2.0) & (t < 2.1), -400.0, 0.0)  # disturbance
    # Cross-coupling: pitch motion disturbs yaw
    cross_coupling = np.where((t >= 0.5) & (t < 0.7), -80.0, 0.0)

    J = 0.05; friction = 0.5

    def sim_single(sp, disturbance):
        kp, ki, kd = 15.0, 2.0, 1.5
        pos = vel = integral = prev_m = 0.0
        positions = []
        for i in range(len(t)):
            e = sp[i] - pos
            integral += ki * e * dt
            integral = np.clip(integral, -5000, 5000)
            d = -(pos - prev_m) / dt
            torque = kp * e + integral + kd * d
            torque = np.clip(torque, -30000, 30000) + disturbance[i]
            vel += (torque - friction * vel) / J * dt
            prev_m = pos
            pos += vel * dt
            positions.append(pos)
        return np.array(positions)

    def sim_cascaded(sp, disturbance):
        kp_o, kp_i, ki_i = 10.0, 80.0, 5.0
        pos = vel = int_i = 0.0
        positions = []
        for i in range(len(t)):
            vel_cmd = np.clip(kp_o * (sp[i] - pos), -500, 500)
            e_i = vel_cmd - vel
            int_i += ki_i * e_i * dt
            int_i = np.clip(int_i, -5000, 5000)
            torque = np.clip(kp_i * e_i + int_i, -30000, 30000) + disturbance[i]
            vel += (torque - friction * vel) / J * dt
            pos += vel * dt
            positions.append(pos)
        return np.array(positions)

    # Single PID — each axis independently, no cross-coupling awareness
    y_single_yaw = sim_single(sp_yaw, dist + cross_coupling)
    y_single_pitch = sim_single(sp_pitch, np.zeros_like(t))

    # Cascaded PID — each axis independently
    y_casc_yaw = sim_cascaded(sp_yaw, dist + cross_coupling)
    y_casc_pitch = sim_cascaded(sp_pitch, np.zeros_like(t))

    # Parallel-Cascaded — both axes cascaded, with cross-coupling compensation
    kp_o, kp_i, ki_i = 10.0, 80.0, 5.0
    pos_y = vel_y = int_iy = 0.0
    pos_p = vel_p = int_ip = 0.0
    yaw_pc = []; pitch_pc = []
    for i in range(len(t)):
        # Yaw axis with cross-coupling compensation
        pitch_rate = vel_p  # known from pitch axis
        cross_comp = -0.3 * pitch_rate  # compensate coupling
        vel_cmd_y = np.clip(kp_o * (sp_yaw[i] - pos_y), -500, 500)
        e_iy = vel_cmd_y - vel_y
        int_iy += ki_i * e_iy * dt
        int_iy = np.clip(int_iy, -5000, 5000)
        torque_y = np.clip(kp_i * e_iy + int_iy + cross_comp, -30000, 30000) \
                   + dist[i] + cross_coupling[i]
        vel_y += (torque_y - friction * vel_y) / J * dt
        pos_y += vel_y * dt
        yaw_pc.append(pos_y)
        # Pitch axis
        vel_cmd_p = np.clip(kp_o * (sp_pitch[i] - pos_p), -500, 500)
        e_ip = vel_cmd_p - vel_p
        int_ip += ki_i * e_ip * dt
        int_ip = np.clip(int_ip, -5000, 5000)
        torque_p = np.clip(kp_i * e_ip + int_ip, -30000, 30000)
        vel_p += (torque_p - friction * vel_p) / J * dt
        pos_p += vel_p * dt
        pitch_pc.append(pos_p)
    yaw_pc = np.array(yaw_pc)
    pitch_pc = np.array(pitch_pc)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Yaw comparison
    ax = axes[0]
    ax.plot(t, sp_yaw, 'k--', linewidth=1, label='Setpoint')
    ax.plot(t, y_single_yaw, 'C3', linewidth=1.5, alpha=0.8, label='Single PID')
    ax.plot(t, y_casc_yaw, 'C0', linewidth=1.5, alpha=0.8, label='Cascaded PID')
    ax.plot(t, yaw_pc, 'C2', linewidth=2, label='Parallel-Cascaded (w/ compensation)')
    ax.axvline(2.0, color='C1', linestyle=':', alpha=0.6)
    ax.annotate('Disturbance', xy=(2.0, 60), fontsize=9, color='C1', ha='center')
    ax.axvline(0.5, color='C4', linestyle=':', alpha=0.4)
    ax.annotate('Cross-coupling\nfrom pitch', xy=(0.6, 30), fontsize=8, color='C4')
    ax.set_ylabel('Yaw angle (deg)')
    ax.set_title('PID Architectures Comparison — Gimbal Dual-Axis Control')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(-15, 130)

    # Pitch comparison
    ax = axes[1]
    ax.plot(t, sp_pitch, 'k--', linewidth=1, label='Setpoint')
    ax.plot(t, y_single_pitch, 'C3', linewidth=1.5, alpha=0.8, label='Single PID')
    ax.plot(t, y_casc_pitch, 'C0', linewidth=1.5, alpha=0.8, label='Cascaded PID')
    ax.plot(t, pitch_pc, 'C2', linewidth=2, label='Parallel-Cascaded')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch angle (deg)')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(-10, 70)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'pid_architectures.pdf'))
    plt.close()
    print('[OK] pid_architectures.pdf')


# ============================================================================
# Run all
# ============================================================================
if __name__ == '__main__':
    print('Generating figures...')
    fig_frequency_spectrum()
    fig_lpf_response()
    fig_lpf_soft_start()
    fig_second_order()
    fig_pid_tuning()
    fig_bode_plot()
    fig_kalman_filter()
    fig_lqr_balance()
    fig_anti_windup()
    fig_cascaded_pid()
    fig_pid_architectures()
    print(f'\nAll figures saved to {OUT}/')

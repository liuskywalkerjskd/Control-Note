#!/usr/bin/env python3
"""
Generate DC motor step response figure (Chinese version): full 2nd-order model vs 1st-order (L≈0) approximation.
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

# --------------------------------------------------------------------------
# Motor parameters
# --------------------------------------------------------------------------
R   = 1.0       # Ohm
L   = 0.5e-3    # Henry (0.5 mH)
Kt  = 0.01      # N·m/A  (torque constant)
Ke  = 0.01      # V·s/rad (back-EMF constant)
J   = 1e-5      # kg·m²
b   = 1e-6      # N·m·s/rad
V   = 12.0      # step voltage input

# Time constants
tau_e = L / R                          # electrical:  L/R
tau_m = J / b                          # mechanical:  J/b
# Effective mechanical time constant under load
# 1st-order: tau_m_eff = (R*J) / (R*b + Kt*Ke)
tau_m_eff = (R * J) / (R * b + Kt * Ke)

print(f"tau_e  = {tau_e*1e6:.1f} µs  ({tau_e*1e3:.3f} ms)")
print(f"tau_m  = {tau_m*1e3:.2f} ms")
print(f"tau_m_eff = {tau_m_eff*1e3:.2f} ms  (1st-order approximation)")

# --------------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------------
t_end = 0.20        # 200 ms
dt    = 1e-6        # 1 µs — fine enough to resolve electrical transient

N = int(t_end / dt) + 1
t = np.linspace(0, t_end, N)

# --------------------------------------------------------------------------
# Full 2nd-order simulation (Euler integration)
# State: x = [i, omega]
# L di/dt = V - R*i - Ke*omega
# J domega/dt = Kt*i - b*omega
# --------------------------------------------------------------------------
i_full   = np.zeros(N)
w_full   = np.zeros(N)

i_cur, w_cur = 0.0, 0.0
for k in range(1, N):
    di = (V - R * i_cur - Ke * w_cur) / L
    dw = (Kt * i_cur - b * w_cur) / J
    i_cur += di * dt
    w_cur += dw * dt
    i_full[k] = i_cur
    w_full[k] = w_cur

rpm_full = w_full * 60.0 / (2.0 * np.pi)

# --------------------------------------------------------------------------
# 1st-order approximation (L ≈ 0)
# i ≈ (V - Ke*omega) / R  (instantaneous)
# J domega/dt = Kt*(V - Ke*omega)/R - b*omega
#            = (Kt*V/R) - (Kt*Ke/R + b)*omega
# Solution: omega(t) = omega_ss * (1 - exp(-t/tau_m_eff))
# --------------------------------------------------------------------------
omega_ss = (Kt * V / R) / (Kt * Ke / R + b)
w_approx = omega_ss * (1.0 - np.exp(-t / tau_m_eff))
rpm_approx = w_approx * 60.0 / (2.0 * np.pi)

# Current from algebraic constraint
i_approx = (V - Ke * w_approx) / R

t_ms = t * 1e3   # convert to milliseconds

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("直流电机阶跃响应：完整模型 vs 简化模型", fontsize=13)

# ---- Left subplot: Full 2nd-order model ----
color_rpm  = '#1f77b4'   # blue
color_curr = '#d62728'   # red

ax1.set_title("完整模型 (二阶)", fontsize=11)
ax1.plot(t_ms, rpm_full, color=color_rpm, label='转速 (RPM)')
ax1.set_xlabel("时间 (ms)")
ax1.set_ylabel("转速 (RPM)", color=color_rpm)
ax1.tick_params(axis='y', labelcolor=color_rpm)

ax1b = ax1.twinx()
ax1b.plot(t_ms, i_full, color=color_curr, linewidth=1.5, linestyle='--', label='电流 (A)')
ax1b.set_ylabel("电流 (A)", color=color_curr)
ax1b.tick_params(axis='y', labelcolor=color_curr)

# Mark tau_e and tau_m_eff with vertical dashed lines
ax1.axvline(tau_e * 1e3, color='gray', linestyle=':', linewidth=1.2,
            label=f'τ_e = {tau_e*1e6:.0f} µs')
ax1.axvline(tau_m_eff * 1e3, color='orange', linestyle=':', linewidth=1.4,
            label=f'τ_m = {tau_m_eff*1e3:.1f} ms')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')

ax1.set_xlim(0, t_ms[-1])
ax1.set_ylim(bottom=0)

# ---- Right subplot: Simplified 1st-order model + overlay ----
ax2.set_title("简化模型 (L ≈ 0)", fontsize=11)
ax2.plot(t_ms, rpm_approx, color='#2ca02c', label='转速 — 简化模型')
ax2.plot(t_ms, rpm_full,   color=color_rpm, linestyle='--', linewidth=1.5,
         label='转速 — 完整模型 (参考)')

ax2.axvline(tau_m_eff * 1e3, color='orange', linestyle=':', linewidth=1.4,
            label=f'τ_m = {tau_m_eff*1e3:.1f} ms')

ax2.set_xlabel("时间 (ms)")
ax2.set_ylabel("转速 (RPM)")
ax2.legend(fontsize=8, loc='center right')
ax2.set_xlim(0, t_ms[-1])
ax2.set_ylim(bottom=0)

# Match y-axis range between subplots for easy comparison
y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1]) * 1.05
ax1.set_ylim(0, y_max)
ax2.set_ylim(0, y_max)

plt.tight_layout()

os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, 'motor_step_response.pdf')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
print("[OK]")

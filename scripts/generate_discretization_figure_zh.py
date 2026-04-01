#!/usr/bin/env python3
"""
Generate a figure comparing ZOH vs Tustin (bilinear) discretization methods (Chinese version)
for a 2nd-order underdamped continuous-time system.
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

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures_zh')

plt.rcParams.update({
    'figure.figsize': (14, 5),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'savefig.dpi': 200,
})

# ── System parameters ────────────────────────────────────────────────────────
omega_n = 10.0   # natural frequency [rad/s]
zeta    = 0.3    # damping ratio (underdamped)
Ts      = 0.05   # sampling period [s]  (~3x Nyquist, deliberately coarse)

# ── Build systems ────────────────────────────────────────────────────────────
sys_c = signal.TransferFunction([omega_n**2],
                                [1, 2*zeta*omega_n, omega_n**2])

# cont2discrete returns (num, den, dt); num may be 2-D — flatten for freqz
def _disc(num, den, dt, **kw):
    b, a, _ = signal.cont2discrete((num, den), dt, **kw)
    return b.flatten(), a.flatten()

b_zoh,    a_zoh    = _disc(sys_c.num, sys_c.den, Ts, method='zoh')
b_tustin, a_tustin = _disc(sys_c.num, sys_c.den, Ts, method='bilinear')
b_prewarp, a_prewarp = _disc(sys_c.num, sys_c.den, Ts,
                              method='bilinear', alpha=omega_n)

# Keep tuple form for dlsim
sys_zoh    = (b_zoh,    a_zoh,    Ts)
sys_tustin = (b_tustin, a_tustin, Ts)
sys_prewarp = (b_prewarp, a_prewarp, Ts)

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3)
fig.suptitle(r'离散化方法对比: ZOH vs Tustin'
             r' ($T_s = 50\,\mathrm{ms}$, $\omega_n = 10\,\mathrm{rad/s}$)',
             fontsize=13, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════════════
# Subplot 1 – Step Response
# ═══════════════════════════════════════════════════════════════════════════════
ax1 = axes[0]

t_end   = 2.0
t_cont  = np.linspace(0, t_end, 2000)
t_c, y_c = signal.step(sys_c, T=t_cont)

# Discrete step responses at sample instants
t_disc = np.arange(0, t_end + Ts, Ts)
n_disc = len(t_disc)

# ZOH — use dlsim with a unit step input
u_step  = np.ones(n_disc)
_, y_zoh   = signal.dlsim(sys_zoh,    u_step, t=t_disc)
_, y_tustin = signal.dlsim(sys_tustin, u_step, t=t_disc)
y_zoh    = y_zoh.flatten()
y_tustin = y_tustin.flatten()

ax1.plot(t_c, y_c, 'k--', lw=1.5, label='连续', zorder=5)

markerline, stemlines, baseline = ax1.stem(
    t_disc, y_zoh, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='ZOH')
stemlines.set_linewidth(1.0)
markerline.set_markersize(4)

markerline2, stemlines2, baseline2 = ax1.stem(
    t_disc, y_tustin, linefmt='C1-', markerfmt='C1s', basefmt=' ', label='Tustin')
stemlines2.set_linewidth(1.0)
markerline2.set_markersize(4)

ax1.set_xlabel('时间 [s]')
ax1.set_ylabel('幅值')
ax1.set_title('阶跃响应')
ax1.legend(fontsize=9)
ax1.set_xlim(0, t_end)

# ═══════════════════════════════════════════════════════════════════════════════
# Subplot 2 – Bode Magnitude
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

f_nyq   = 0.5 / Ts                          # Nyquist frequency [Hz]
w_nyq   = np.pi / Ts                        # [rad/s]
w_cont  = np.logspace(-1, np.log10(w_nyq * 1.05), 500)

# Continuous Bode
w_c, H_c = signal.freqs(sys_c.num, sys_c.den, worN=w_cont)
mag_c = 20 * np.log10(np.abs(H_c))

# ZOH digital Bode (w mapped to digital freq)
w_d    = np.linspace(1e-3, np.pi, 500)      # digital frequency [rad/sample]
w_hz   = w_d / Ts                           # [rad/s]
_, H_zoh     = signal.freqz(b_zoh,    a_zoh,    worN=w_d)
_, H_tustin  = signal.freqz(b_tustin, a_tustin, worN=w_d)
_, H_prewarp = signal.freqz(b_prewarp, a_prewarp, worN=w_d)
mag_zoh     = 20 * np.log10(np.abs(H_zoh))
mag_tustin  = 20 * np.log10(np.abs(H_tustin))
mag_prewarp = 20 * np.log10(np.abs(H_prewarp))

ax2.semilogx(w_cont,  mag_c,      'k',   lw=2,   label='连续')
ax2.semilogx(w_hz,    mag_zoh,    'C0',  lw=2,   label='ZOH')
ax2.semilogx(w_hz,    mag_tustin, 'C1',  lw=2,   label='Tustin')
ax2.semilogx(w_hz,    mag_prewarp,'g--', lw=1.5, label='Tustin (预翘曲)')

ax2.axvline(w_nyq, color='gray', ls=':', lw=1.2, label=f'奈奎斯特 ({w_nyq:.0f} rad/s)')

ax2.set_xlabel('频率 [rad/s]')
ax2.set_ylabel('幅值 [dB]')
ax2.set_title('波特幅值图')
ax2.legend(fontsize=8)
ax2.set_xlim(w_cont[0], w_nyq * 1.05)
ax2.set_ylim(-60, 10)

# ═══════════════════════════════════════════════════════════════════════════════
# Subplot 3 – Pole-Zero Map (z-plane)
# ═══════════════════════════════════════════════════════════════════════════════
ax3 = axes[2]

# Unit circle
theta = np.linspace(0, 2*np.pi, 300)
ax3.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, alpha=0.5, label='单位圆')
ax3.axhline(0, color='k', lw=0.5, alpha=0.3)
ax3.axvline(0, color='k', lw=0.5, alpha=0.3)

# Continuous poles  s = -ζωn ± jωn√(1-ζ²)
s_poles = np.roots(sys_c.den)
z_exact = np.exp(s_poles * Ts)   # exact mapping z = e^{sT}

# ZOH poles
zoh_poles    = np.roots(a_zoh)
tustin_poles = np.roots(a_tustin)

ax3.plot(z_exact.real,    z_exact.imag,    'kx', ms=10, mew=2.5,
         label=r'$z = e^{sT}$ (精确)')
ax3.plot(zoh_poles.real,    zoh_poles.imag,    'C0o', ms=8, mfc='none', mew=2,
         label='ZOH 极点')
ax3.plot(tustin_poles.real, tustin_poles.imag, 'C1o', ms=8, mfc='none', mew=2,
         label='Tustin 极点')

# Note: for ZOH the poles equal e^{sT} exactly, so they should overlap with kx
ax3.set_xlabel('实部')
ax3.set_ylabel('虚部')
ax3.set_title('零极点图 (z 平面)')
ax3.legend(fontsize=8, loc='upper left')
ax3.set_aspect('equal')
margin = 0.15
ax3.set_xlim(-1 - margin, 1 + margin)
ax3.set_ylim(-1 - margin, 1 + margin)

# ── Save ─────────────────────────────────────────────────────────────────────
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, 'discretization_comparison.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f'Saved: {out_path}')

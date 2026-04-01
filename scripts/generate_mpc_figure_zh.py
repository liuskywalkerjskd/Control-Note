#!/usr/bin/env python3
"""
Generate MPC Receding Horizon figure (Chinese version) illustrating the
"plan ahead, execute one step, replan" concept on a double-integrator
(position tracking, sinusoidal reference).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from zh_config import apply_zh
apply_zh()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures_zh')

plt.rcParams.update({
    'figure.figsize': (14, 5),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'savefig.dpi': 200,
})

# ── simulation parameters ──────────────────────────────────────────────────────
dt   = 1.0          # time step (s)
N    = 10           # prediction horizon
T    = 25           # total simulation steps to show
U_MAX = 2.0         # acceleration limit [m/s²]

# reference: sinusoidal position
t_full = np.arange(0, T + N + 2, dt)
ref    = 3.0 * np.sin(0.25 * t_full)

# ── double-integrator dynamics: x = [pos, vel] ─────────────────────────────────
A = np.array([[1, dt], [0, 1]])
B = np.array([[0.5 * dt**2], [dt]])

def step(x, u):
    u = np.clip(u, -U_MAX, U_MAX)
    return A @ x + B.flatten() * u

def compute_mpc_input(x0, ref_traj):
    """
    Simple greedy receding-horizon controller:
    at each horizon step choose u in [-U_MAX, U_MAX] that minimises
    (position - reference)^2 using a bang-bang + proportional heuristic.
    Returns the sequence of optimal positions and the first control action.
    """
    x    = x0.copy()
    pred_pos = [x[0]]
    first_u  = None
    for i in range(N):
        err = ref_traj[i + 1] - x[0]
        vel = x[1]
        # proportional feedback + derivative damping
        u_raw = 1.5 * err - 1.2 * vel
        u = np.clip(u_raw, -U_MAX, U_MAX)
        if first_u is None:
            first_u = u
        x = step(x, u)
        pred_pos.append(x[0])
    return np.array(pred_pos), first_u

# ── run closed-loop simulation to build the "past" trajectory ──────────────────
np.random.seed(0)
x_state = np.array([0.0, 0.0])   # initial state: pos=0, vel=0
history_pos = [x_state[0]]

for k in range(T):
    ref_horizon = ref[k: k + N + 2]
    _, u0 = compute_mpc_input(x_state, ref_horizon)
    x_state = step(x_state, u0)
    history_pos.append(x_state[0])

history_pos = np.array(history_pos)   # shape (T+1,)

# ── rebuild states at snapshot times ──────────────────────────────────────────
# We need to re-integrate up to each snapshot to get the velocity right.
def get_state_at(k):
    x = np.array([0.0, 0.0])
    for i in range(k):
        ref_h = ref[i: i + N + 2]
        _, u0 = compute_mpc_input(x, ref_h)
        x = step(x, u0)
    return x

snapshot_steps = [0, 5, 10]
titles = ["步骤 $k=0$", "步骤 $k=5$", "步骤 $k=10$"]
pred_colors = ['#2ca02c', '#ff7f0e', '#9467bd']   # green, orange, purple

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, sharey=True)
fig.suptitle("MPC 滚动时域：向前规划，执行一步，重新规划",
             fontsize=13, fontweight='bold', y=1.01)

for col, (k_now, title, pcolor) in enumerate(zip(snapshot_steps, titles, pred_colors)):
    ax = axes[col]

    x_k = get_state_at(k_now)

    # predicted trajectory from k_now
    ref_horizon = ref[k_now: k_now + N + 2]
    pred_pos, u0 = compute_mpc_input(x_k, ref_horizon)
    pred_times   = np.arange(k_now, k_now + N + 1)

    # ── reference (dashed black, shown over full visible window) ──────────────
    show_start = max(0, k_now - 6)
    show_end   = k_now + N + 1
    t_show     = np.arange(show_start, show_end + 1)
    ax.plot(t_show, ref[show_start: show_end + 1],
            'k--', lw=1.5, label='参考轨迹', zorder=2)

    # ── past actual trajectory (solid blue) ───────────────────────────────────
    if k_now > 0:
        past_t   = np.arange(show_start, k_now + 1)
        past_pos = history_pos[show_start: k_now + 1]
        ax.plot(past_t, past_pos, 'b-', lw=2.5, label='实际(历史)', zorder=3)
        # shade the past region
        ax.axvspan(show_start - 0.5, k_now, color='lightblue', alpha=0.15, zorder=0)

    # ── predicted trajectory (colored line + shaded band) ─────────────────────
    ax.plot(pred_times, pred_pos, '-o', color=pcolor, ms=5,
            lw=2, label=f'预测 (N={N})', zorder=4)
    # light fill under prediction horizon
    ax.fill_between(pred_times, pred_pos - 0.3, pred_pos + 0.3,
                    color=pcolor, alpha=0.15, zorder=1)
    # shade the prediction horizon region
    ax.axvspan(k_now, k_now + N, color='lightyellow', alpha=0.25, zorder=0)

    # ── highlight the first applied control action (first predicted point) ─────
    ax.plot(k_now, pred_pos[0], 'o', color=pcolor, ms=12,
            mec='red', mew=2.5, zorder=6, label='仅执行 $u_0$')
    ax.plot(k_now + 1, pred_pos[1], 's', color=pcolor, ms=8,
            mec=pcolor, mew=1.5, zorder=5, alpha=0.8)

    # ── annotation: "apply u[0] only" ─────────────────────────────────────────
    y_top = ax.get_ylim()[1] if col > 0 else 4.2
    ax.annotate("仅执行 $u_0$",
                xy=(k_now, pred_pos[0]),
                xytext=(k_now + 1.2, pred_pos[0] + 1.5),
                fontsize=9, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                zorder=7)

    # ── annotation: "replan next step" (at right edge of horizon) ─────────────
    ax.annotate("下一步\n重新规划",
                xy=(k_now + N, pred_pos[-1]),
                xytext=(k_now + N - 2.5, pred_pos[-1] - 1.8),
                fontsize=8.5, color='#555555',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
                zorder=7)

    # ── vertical dashed line at current step ──────────────────────────────────
    ax.axvline(k_now, color='red', lw=1.5, ls='--', alpha=0.8, zorder=5)

    # ── u_max indicator (acceleration limit band) ──────────────────────────────
    # Show max reachable position offset at next step (visual hint)
    ax.annotate(f'$|u| \\leq {U_MAX}$ m/s²',
                xy=(k_now + 0.3, -3.8), fontsize=8, color='gray',
                style='italic')

    # ── region labels ──────────────────────────────────────────────────────────
    if k_now > 0:
        ax.text((show_start + k_now) / 2, 3.7, '历史',
                ha='center', va='bottom', fontsize=9,
                color='steelblue', fontstyle='italic')
    ax.text(k_now + N / 2, 3.7, f'预测时域\n$(N={N})$',
            ha='center', va='bottom', fontsize=9,
            color=pcolor, fontstyle='italic')

    # ── axes formatting ────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('时间步 $k$')
    if col == 0:
        ax.set_ylabel('位置 (m)')
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlim(show_start - 0.5, k_now + N + 0.5)

    # ── legend (only for first subplot to avoid clutter) ──────────────────────
    if col == 0:
        legend_elements = [
            Line2D([0], [0], ls='--', color='black',  lw=1.5, label='参考轨迹'),
            Line2D([0], [0], ls='-',  color='blue',   lw=2.5, label='实际(历史)'),
            Line2D([0], [0], ls='-',  color=pcolor,   lw=2,   label=f'预测 (N={N})'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=pcolor,
                   markersize=10, mec='red', mew=2.5,           label='仅执行 $u_0$'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8.5,
                  framealpha=0.85)

plt.tight_layout()

os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, 'mpc_horizon.pdf')
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")

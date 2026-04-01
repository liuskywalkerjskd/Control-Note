"""
Generate trajectory profiles figure (Chinese version): Trapezoidal vs S-Curve (7-phase jerk-limited).

Parameters:
    distance = 90 deg
    v_max    = 200 deg/s
    a_max    = 800 deg/s²
    j_max    = 10000 deg/s³
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from zh_config import apply_zh
apply_zh()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.figsize': (8, 4.5),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'savefig.dpi': 200,
})

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
D      = 90.0     # degrees
v_max  = 200.0    # deg/s
a_max  = 800.0    # deg/s²
j_max  = 10000.0  # deg/s³
dt     = 0.0001   # s

# ---------------------------------------------------------------------------
# Trapezoidal Profile (3-phase: accel / cruise / decel)
# ---------------------------------------------------------------------------
def trapezoidal_profile(D, v_max, a_max, dt=0.0001):
    """
    Phase 1: accelerate from 0 to v_peak at a_max
    Phase 2: cruise at v_peak
    Phase 3: decelerate from v_peak to 0 at -a_max

    v_peak may be less than v_max if distance is short.
    Returns t, pos, vel, acc arrays and phase boundary times.
    """
    # Ramp distance (one ramp: 0 -> v_peak)
    # v_peak² = a_max * d_ramp  =>  d_ramp = v_peak²/(2*a_max)
    # Two ramps: d_ramps = v_peak²/a_max
    # If v_peak²/a_max > D, triangle profile: v_peak = sqrt(a_max*D)
    v_peak = min(v_max, np.sqrt(a_max * D))

    t_accel  = v_peak / a_max
    d_accel  = 0.5 * a_max * t_accel**2          # each ramp
    d_cruise = D - 2.0 * d_accel
    t_cruise = d_cruise / v_peak if v_peak > 0 else 0.0
    t_decel  = t_accel
    T_total  = t_accel + t_cruise + t_decel

    # Phase boundaries
    t1 = t_accel
    t2 = t_accel + t_cruise
    t3 = T_total

    t_arr = np.arange(0.0, T_total + dt, dt)
    pos = np.zeros_like(t_arr)
    vel = np.zeros_like(t_arr)
    acc = np.zeros_like(t_arr)

    for i, t in enumerate(t_arr):
        if t <= t1:                          # Phase 1: accelerate
            a = a_max
            v = a_max * t
            p = 0.5 * a_max * t**2
        elif t <= t2:                        # Phase 2: cruise
            tau = t - t1
            a = 0.0
            v = v_peak
            p = d_accel + v_peak * tau
        else:                                # Phase 3: decelerate
            tau = t - t2
            a = -a_max
            v = v_peak - a_max * tau
            p = d_accel + d_cruise + v_peak * tau - 0.5 * a_max * tau**2
        acc[i] = a
        vel[i] = max(v, 0.0)
        pos[i] = min(max(p, 0.0), D)

    return t_arr, pos, vel, acc, [0.0, t1, t2, t3]


# ---------------------------------------------------------------------------
# 7-Phase S-Curve (jerk-limited) Profile
# ---------------------------------------------------------------------------
def scurve_profile(D, v_max, a_max, j_max, dt=0.0001):
    """
    Classic 7-phase jerk-limited profile:
      1: +j_max  (jerk up)      -> a ramps 0 -> a_max
      2: j=0     (const accel)  -> a stays at a_max
      3: -j_max  (jerk down)    -> a ramps a_max -> 0   (v = v_peak)
      4: j=0     (cruise)       -> constant velocity
      5: -j_max  (jerk down)    -> a ramps 0 -> -a_max
      6: j=0     (const decel)  -> a stays at -a_max
      7: +j_max  (jerk up)      -> a ramps -a_max -> 0  (v = 0)

    v_peak may be limited by distance.
    a_peak may be limited if distance is very short.
    """
    # Time to reach a_max with j_max
    t_j = a_max / j_max          # jerk phase duration
    # Velocity gained during single jerk phase (0->a_max): 0.5*j_max*t_j²
    dv_jerk = 0.5 * j_max * t_j**2   # = a_max²/(2*j_max)

    # v_peak achievable with full a_max ramp:
    # One full accel segment (phases 1-2-3): v = 2*dv_jerk + a_max*t_a2
    # Total distance for full accel+decel (symmetric):
    #   d = 2*(dist_phases_123 + dist_phases_567)
    # For simplicity, solve iteratively or use closed-form.

    # --- Determine v_peak ---
    # Full triangular (no phase-2/6): v_peak_tri = j_max * t_j² = a_max²/j_max  (both ramps)
    # Actually v_peak_tri = 2*dv_jerk
    v_peak_tri = 2.0 * dv_jerk   # max velocity if no const-accel phase

    # Distance for a pure 7-phase accel+decel (no cruise, no const-accel):
    # d = 2 * (v_peak_tri/2) * (2*t_j) ... geometry of the jerk segments
    # More precisely: d_half_accel (phases 1-3) when a_max not reached:
    #   phases 1 and 3 each: t_j, a goes 0->a_p->0 where a_p=j_max*t_j
    #   v at end of phase1: 0.5*j_max*t_j²
    #   v at end of phase3: v_peak = 2*0.5*j_max*t_j² = j_max*t_j²
    # d_accel_tri = integral(v dt) over phases 1+3 (no phase 2)
    #   = (1/6)*j_max*t_j³ + (v_peak - 1/6*j_max*t_j³)
    # Let's compute numerically for clarity.

    def compute_scurve(v_peak, a_peak):
        """
        Compute S-curve phases for given v_peak and a_peak (<=a_max).
        Returns phase durations [t1..t7] and distances.
        """
        if a_peak < 1e-9:
            return None
        tj = a_peak / j_max          # time for jerk phase
        dv_j = 0.5 * j_max * tj**2  # vel gained in each jerk phase

        # Const-accel phase duration to reach v_peak/2 from dv_j:
        # v at end of ph1 = dv_j
        # v at end of ph2 = dv_j + a_peak*t2 = v_peak/2 => t2 = (v_peak/2 - dv_j)/a_peak
        t2 = max((v_peak / 2.0 - dv_j) / a_peak, 0.0)

        # Phase durations (symmetric accel/decel)
        t1 = tj; t3 = tj; t5 = tj; t7 = tj
        t6 = t2

        # Distance calculations
        # Phase 1: a=j_max*t, v=0.5*j_max*t², p = (1/6)*j_max*t³
        d1 = (1.0/6.0) * j_max * t1**3
        v1 = dv_j   # vel at end of ph1

        # Phase 2: const a=a_peak
        d2 = v1 * t2 + 0.5 * a_peak * t2**2
        v2 = v1 + a_peak * t2   # vel at end of ph2

        # Phase 3: a goes from a_peak to 0 (jerk = -j_max)
        # v(t) = v2 + a_peak*t - 0.5*j_max*t²
        # at t=t3: v = v2 + a_peak*tj - 0.5*j_max*tj² = v2 + dv_j (since a_peak=j_max*tj)
        d3 = v2 * t3 + 0.5 * a_peak * t3**2 - (1.0/6.0) * j_max * t3**3
        v3 = v2 + dv_j   # = v_peak

        # Accel distance (half)
        d_accel = d1 + d2 + d3
        # By symmetry, decel distance = d_accel

        return d_accel, t1, t2, t3, v3

    # Try full a_max first
    res = compute_scurve(v_max, a_max)
    if res is not None:
        d_accel, t1, t2, t3, v3 = res
        d_needed = 2.0 * d_accel
        if d_needed > D:
            # Need to reduce v_peak (no cruise phase)
            # Binary search for v_peak such that 2*d_accel = D
            lo, hi = 0.0, v_max
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                # For given v_peak, determine a_peak
                # If v_peak <= 2*dv_jerk_full (i.e., a_max not reached):
                v_peak_tri_full = a_max**2 / j_max
                if mid <= v_peak_tri_full:
                    # a_peak = sqrt(j_max * mid) (since v_peak = j_max*t_j²,  a_peak=j_max*t_j)
                    a_peak_try = np.sqrt(j_max * mid)
                else:
                    a_peak_try = a_max
                r = compute_scurve(mid, a_peak_try)
                if r is None:
                    hi = mid; continue
                d_a = r[0]
                if 2.0 * d_a < D:
                    lo = mid
                else:
                    hi = mid
            v_peak = 0.5 * (lo + hi)
            v_peak_tri_full = a_max**2 / j_max
            if v_peak <= v_peak_tri_full:
                a_peak = np.sqrt(j_max * v_peak)
            else:
                a_peak = a_max
            t_cruise_v = 0.0
        else:
            v_peak = v_max
            a_peak = a_max
            d_cruise_dist = D - d_needed
            t_cruise_v = d_cruise_dist / v_peak
    else:
        v_peak = v_max; a_peak = a_max; t_cruise_v = 0.0

    # Recompute final phase durations
    res = compute_scurve(v_peak, a_peak)
    d_accel, t1_d, t2_d, t3_d, _ = res
    # Recompute t_cruise properly
    d_cruise_dist = D - 2.0 * d_accel
    t_cruise_v = max(d_cruise_dist / v_peak, 0.0)

    tj = a_peak / j_max
    t1 = tj; t2 = t2_d; t3 = tj
    t4 = t_cruise_v
    t5 = tj; t6 = t2_d; t7 = tj

    # Cumulative phase end times
    T = [0]
    for ti in [t1, t2, t3, t4, t5, t6, t7]:
        T.append(T[-1] + ti)
    T_total = T[-1]

    t_arr = np.arange(0.0, T_total + dt, dt)
    pos_arr = np.zeros_like(t_arr)
    vel_arr = np.zeros_like(t_arr)
    acc_arr = np.zeros_like(t_arr)

    dv_j = 0.5 * j_max * tj**2

    for i, t in enumerate(t_arr):
        if t <= T[1]:                        # Phase 1: jerk up
            tau = t - T[0]
            j = j_max
            a = j_max * tau
            v = 0.5 * j_max * tau**2
            p = (1.0/6.0) * j_max * tau**3

        elif t <= T[2]:                      # Phase 2: const accel
            tau = t - T[1]
            j = 0.0
            a = a_peak
            v_t1 = dv_j
            v = v_t1 + a_peak * tau
            d1 = (1.0/6.0) * j_max * t1**3
            p = d1 + v_t1 * tau + 0.5 * a_peak * tau**2

        elif t <= T[3]:                      # Phase 3: jerk down (accel->0)
            tau = t - T[2]
            j = -j_max
            a = a_peak - j_max * tau
            v_t2 = dv_j + a_peak * t2
            d1 = (1.0/6.0) * j_max * t1**3
            d2_seg = dv_j * t2 + 0.5 * a_peak * t2**2
            v = v_t2 + a_peak * tau - 0.5 * j_max * tau**2
            p = d1 + d2_seg + v_t2 * tau + 0.5 * a_peak * tau**2 - (1.0/6.0) * j_max * tau**3

        elif t <= T[4]:                      # Phase 4: cruise
            tau = t - T[3]
            j = 0.0; a = 0.0
            v = v_peak
            p = d_accel + v_peak * tau

        elif t <= T[5]:                      # Phase 5: jerk down (0->-a_peak)
            tau = t - T[4]
            j = -j_max
            a = -j_max * tau
            v = v_peak - 0.5 * j_max * tau**2
            p = d_accel + d_cruise_dist + v_peak * tau - (1.0/6.0) * j_max * tau**3

        elif t <= T[6]:                      # Phase 6: const decel
            tau = t - T[5]
            j = 0.0
            a = -a_peak
            v_t5 = v_peak - dv_j
            d5 = v_peak * t5 - (1.0/6.0) * j_max * t5**3
            p_ph5 = d_accel + d_cruise_dist + d5
            v = v_t5 - a_peak * tau
            p = p_ph5 + v_t5 * tau - 0.5 * a_peak * tau**2

        else:                                # Phase 7: jerk up (decel->0)
            tau = t - T[6]
            j = j_max
            a = -a_peak + j_max * tau
            v_t5 = v_peak - dv_j
            d5 = v_peak * t5 - (1.0/6.0) * j_max * t5**3
            v_t6 = v_t5 - a_peak * t6
            d6 = v_t5 * t6 - 0.5 * a_peak * t6**2
            p_ph6 = d_accel + d_cruise_dist + d5 + d6
            v = v_t6 - a_peak * tau + 0.5 * j_max * tau**2
            p = p_ph6 + v_t6 * tau - 0.5 * a_peak * tau**2 + (1.0/6.0) * j_max * tau**3

        acc_arr[i] = a
        vel_arr[i] = max(v, 0.0)
        pos_arr[i] = min(max(p, 0.0), D)

    return t_arr, pos_arr, vel_arr, acc_arr, T


# ---------------------------------------------------------------------------
# Compute profiles
# ---------------------------------------------------------------------------
t_trap, pos_trap, vel_trap, acc_trap, phases_trap = trapezoidal_profile(D, v_max, a_max, dt)
t_scurve, pos_scurve, vel_scurve, acc_scurve, phases_scurve = scurve_profile(D, v_max, a_max, j_max, dt)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(12, 8))
fig.suptitle("运动曲线：梯形速度曲线 vs S形速度曲线", fontsize=13, fontweight='bold', y=0.98)

color_trap   = 'C0'   # blue
color_scurve = 'C1'   # orange

# Phase shading colors (alternating)
trap_phase_colors  = ['#DDEEFF', '#FFEEDD', '#DDEEFF']
scurve_phase_colors = ['#FFF0DD', '#FFE5CC', '#FFF0DD', '#FFEEDD',
                       '#FFF0DD', '#FFE5CC', '#FFF0DD']

trap_phase_labels  = ['加速', '匀速', '减速']
scurve_phase_labels = ['加加速度↑', '恒加速', '加加速度↓', '匀速',
                       '加加速度↓', '恒减速', '加加速度↑']

# Row data: (data_y, ylabel, title_suffix)
rows = [
    (pos_trap,  pos_scurve,  r'$\theta(t)$ [deg]',            r'位置 $\theta(t)$'),
    (vel_trap,  vel_scurve,  r'$\dot\theta$ [°/s]',           r'速度 $\dot\theta(t)$'),
    (acc_trap,  acc_scurve,  r'$\ddot\theta$ [°/s²]',         r'加速度 $\ddot\theta(t)$'),
]

col_titles = ['梯形速度曲线', 'S形速度曲线 (7段加加速度限制)']

for col in range(2):
    for row in range(3):
        ax = axes[row, col]
        data_trap, data_sc, ylabel, row_title = rows[row][0], rows[row][1], rows[row][2], rows[row][3]

        if col == 0:
            t_plot = t_trap
            data   = data_trap
            color  = color_trap
            phases = phases_trap
            phase_colors = trap_phase_colors
            phase_labels = trap_phase_labels
        else:
            t_plot = t_scurve
            data   = data_sc
            color  = color_scurve
            phases = phases_scurve
            phase_colors = scurve_phase_colors
            phase_labels = scurve_phase_labels

        # Shade phases
        y_min_ax = data.min() - 0.05 * (data.max() - data.min() + 1)
        y_max_ax = data.max() + 0.08 * (data.max() - data.min() + 1)
        for pi in range(len(phases) - 1):
            ax.axvspan(phases[pi], phases[pi+1],
                       color=phase_colors[pi % len(phase_colors)],
                       alpha=0.35, zorder=0)
            # Phase label at top of plot (only for velocity row to avoid clutter)
            if row == 1:
                mid_t = 0.5 * (phases[pi] + phases[pi+1])
                ax.text(mid_t, y_max_ax * 0.92, phase_labels[pi],
                        ha='center', va='top', fontsize=7.5, color='#555555',
                        zorder=5)

        ax.plot(t_plot, data, color=color, linewidth=2, zorder=3)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(t_plot[0], t_plot[-1])

        # Add phase vertical lines
        for pb in phases[1:-1]:
            ax.axvline(pb, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        if row == 0:
            ax.set_title(col_titles[col], fontsize=11, fontweight='bold', pad=6)
        if row == 2:
            ax.set_xlabel('时间 [s]', fontsize=10)

# Align ylims for same rows across columns
for row in range(3):
    y_mins = [axes[row, c].get_ylim()[0] for c in range(2)]
    y_maxs = [axes[row, c].get_ylim()[1] for c in range(2)]
    for c in range(2):
        axes[row, c].set_ylim(min(y_mins), max(y_maxs))

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures_zh')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'trajectory_profiles.pdf')
plt.savefig(out_path, bbox_inches='tight')
plt.close()

print(f"Saved to: {os.path.abspath(out_path)}")
print("[OK]")

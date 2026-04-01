"""
Generate Bezier curves and cubic spline interpolation comparison figure (Chinese).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import os
from zh_config import apply_zh

apply_zh()

# ---------------------------------------------------------------------------
# Cubic Bezier
# ---------------------------------------------------------------------------
def bezier_cubic(P0, P1, P2, P3, t):
    u = 1 - t
    return u**3*P0 + 3*u**2*t*P1 + 3*u*t**2*P2 + t**3*P3

def bezier_tangent(P0, P1, P2, P3, t):
    u = 1 - t
    return 3*(u**2*(P1-P0) + 2*u*t*(P2-P1) + t**2*(P3-P2))

P0 = np.array([0.0, 0.0])
P1 = np.array([1.0, 2.5])
P2 = np.array([3.0, 3.0])
P3 = np.array([4.0, 1.0])

t = np.linspace(0, 1, 200)
curve = np.array([bezier_cubic(P0, P1, P2, P3, ti) for ti in t])

# ---------------------------------------------------------------------------
# Cubic Spline
# ---------------------------------------------------------------------------
waypoints_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
waypoints_y = np.array([0.0, 2.0, 1.5, 3.0, 1.0])
cs = CubicSpline(waypoints_x, waypoints_y)
xs = np.linspace(0, 4, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# --- Left: Bezier ---
ax1.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2.5, zorder=3,
         label='三次 B\u00e9zier 曲线')
ctrl = np.array([P0, P1, P2, P3])
ax1.plot(ctrl[:, 0], ctrl[:, 1], 'k--', alpha=0.4, linewidth=1, label='控制多边形')
ax1.plot(ctrl[:, 0], ctrl[:, 1], 'ro', markersize=8, zorder=4)
labels = ['$P_0$', '$P_1$', '$P_2$', '$P_3$']
offsets = [(-0.15, -0.3), (-0.15, 0.2), (0.1, 0.2), (0.1, -0.3)]
for pt, lbl, off in zip(ctrl, labels, offsets):
    ax1.annotate(lbl, pt, textcoords='offset points', xytext=(off[0]*60, off[1]*60),
                fontsize=13, fontweight='bold', color='red')
for ti, color in [(0.0, '#2ca02c'), (1.0, '#d62728')]:
    pos = bezier_cubic(P0, P1, P2, P3, ti)
    tan = bezier_tangent(P0, P1, P2, P3, ti)
    tan = tan / np.linalg.norm(tan) * 0.8
    ax1.annotate('', xy=pos+tan, xytext=pos,
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('三次 B\u00e9zier 曲线')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_aspect('equal', adjustable='datalim')

# --- Right: Spline vs Bezier ---
ax2.plot(xs, cs(xs), 'b-', linewidth=2.5, label='三次样条', zorder=3)
ax2.plot(waypoints_x, waypoints_y, 'ro', markersize=8, zorder=4, label='路径点')

P0b = np.array([waypoints_x[0], waypoints_y[0]])
P3b = np.array([waypoints_x[-1], waypoints_y[-1]])
P1b = np.array([1.0, 3.5])
P2b = np.array([3.0, 3.5])
curve_b = np.array([bezier_cubic(P0b, P1b, P2b, P3b, ti) for ti in t])
ax2.plot(curve_b[:, 0], curve_b[:, 1], '--', color='#ff7f0e', linewidth=2,
         label='B\u00e9zier（相同端点）', alpha=0.8)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('三次样条 vs B\u00e9zier')
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'figures_zh', 'bezier_spline.pdf')
plt.savefig(out, bbox_inches='tight')
print(f'Saved {out}')
plt.close()

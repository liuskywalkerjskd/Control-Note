#!/usr/bin/env python3
"""
Generate grid-world DP illustration for the Bellman equation section.
Shows: (1) the grid with costs, (2) the cost-to-go table filled backward,
(3) the optimal path highlighted.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'figures')

# 5x5 grid with random-ish costs
costs = np.array([
    [1, 3, 1, 2, 4],
    [2, 1, 5, 1, 2],
    [4, 2, 1, 3, 1],
    [1, 5, 2, 1, 3],
    [3, 1, 4, 2, 1],
])
n = 5

# Compute cost-to-go (backward DP)
V = np.full((n, n), np.inf)
V[n-1, n-1] = costs[n-1, n-1]

# Fill backward: can only go right or down
for i in range(n-1, -1, -1):
    for j in range(n-1, -1, -1):
        if i == n-1 and j == n-1:
            continue
        options = []
        if i + 1 < n:
            options.append(V[i+1, j])
        if j + 1 < n:
            options.append(V[i, j+1])
        V[i, j] = costs[i, j] + min(options)

# Trace optimal path
path = [(0, 0)]
i, j = 0, 0
while (i, j) != (n-1, n-1):
    go_down = V[i+1, j] if i+1 < n else np.inf
    go_right = V[i, j+1] if j+1 < n else np.inf
    if go_down <= go_right:
        i += 1
    else:
        j += 1
    path.append((i, j))

# ============================================================================
# Draw figure with 3 panels
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

for ax_idx, (ax, title, show_cost, show_V, show_path) in enumerate(zip(
    axes,
    ['(a) Grid with step costs c(i,j)',
     '(b) Cost-to-go V(i,j) filled backward',
     '(c) Optimal path (Bellman policy)'],
    [True, False, True],
    [False, True, True],
    [False, False, True]
)):
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'j={j}' for j in range(n)], fontsize=8)
    ax.set_yticklabels([f'i={i}' for i in range(n)], fontsize=8)

    for i in range(n):
        for j in range(n):
            # Cell background
            if show_path and (i, j) in path:
                color = '#FFE082' if (i,j) != (0,0) and (i,j) != (n-1,n-1) else '#81C784'
                if (i,j) == (0,0):
                    color = '#64B5F6'
                if (i,j) == (n-1, n-1):
                    color = '#EF5350'
            else:
                color = '#F5F5F5'

            rect = patches.FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor='#757575', linewidth=1.2
            )
            ax.add_patch(rect)

            # Text
            if show_cost and not show_V:
                ax.text(j, i, f'{costs[i,j]}', ha='center', va='center',
                       fontsize=16, fontweight='bold', color='#333333')
            elif show_V and not show_cost:
                ax.text(j, i, f'{int(V[i,j])}', ha='center', va='center',
                       fontsize=14, fontweight='bold', color='#1565C0')
            elif show_V and show_cost:
                ax.text(j, i - 0.15, f'{int(V[i,j])}', ha='center', va='center',
                       fontsize=14, fontweight='bold', color='#1565C0')
                ax.text(j, i + 0.22, f'c={costs[i,j]}', ha='center', va='center',
                       fontsize=8, color='#888888')

    # Draw optimal path arrows
    if show_path:
        for k in range(len(path) - 1):
            i1, j1 = path[k]
            i2, j2 = path[k+1]
            dx = (j2 - j1) * 0.55
            dy = (i2 - i1) * 0.55
            ax.annotate('', xy=(j1 + dx + (j2-j1)*0.15, i1 + dy + (i2-i1)*0.15),
                       xytext=(j1 + (j2-j1)*0.15, i1 + (i2-i1)*0.15),
                       arrowprops=dict(arrowstyle='->', color='#D32F2F',
                                      linewidth=2.5, mutation_scale=15))
        # Labels
        ax.text(0, -0.42, 'START', ha='center', va='bottom', fontsize=7,
               fontweight='bold', color='#1565C0')
        ax.text(n-1, n-1+0.42, 'GOAL', ha='center', va='top', fontsize=7,
               fontweight='bold', color='#C62828')

    # Draw direction hints on first panel
    if ax_idx == 0:
        ax.annotate('', xy=(1.4, 0), xytext=(0.6, 0),
                   arrowprops=dict(arrowstyle='->', color='#888', linewidth=1.5))
        ax.annotate('', xy=(0, 1.4), xytext=(0, 0.6),
                   arrowprops=dict(arrowstyle='->', color='#888', linewidth=1.5))
        ax.text(1.0, -0.35, 'right', ha='center', fontsize=7, color='#888')
        ax.text(-0.35, 1.0, 'down', ha='center', fontsize=7, color='#888', rotation=90)

    # Panel (b): draw backward fill arrows
    if ax_idx == 1:
        # Show a few backward arrows to indicate fill direction
        ax.annotate('fill\nbackward', xy=(3.5, 3.5), fontsize=8, color='#C62828',
                   ha='center', fontweight='bold')
        ax.annotate('', xy=(2.5, 2.5), xytext=(3.3, 3.3),
                   arrowprops=dict(arrowstyle='->', color='#C62828',
                                  linewidth=1.5, linestyle='dashed'))

ax.grid(False)
for ax in axes:
    ax.grid(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'gridworld_dp.pdf'), dpi=200, bbox_inches='tight')
plt.close()
print('[OK] gridworld_dp.pdf')
print(f'Optimal path: {path}')
print(f'Optimal cost: {int(V[0,0])}')

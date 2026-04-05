"""Generate figures for the Attitude and Rotation chapter (EN + ZH)."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec

# ── Localised strings ────────────────────────────────────────
S = {
    'en': {
        'rh_title': 'Right-Handed Frame',
        'lh_title': 'Left-Handed Frame',
        'ned': 'NED (Aerospace)',
        'enu': 'ENU (Robotics/ROS)',
        'north': 'N', 'east': 'E', 'down': 'D', 'up': 'U',
        'world': 'World Frame', 'body': 'Body Frame',
        'world_body_title': 'World Frame vs Body Frame',
        'active_title': 'Active Rotation (rotate the vector)',
        'passive_title': 'Passive Rotation (change the frame)',
        'original': 'Original', 'rotated': 'Rotated',
        'gimbal_title': 'Gimbal Lock: Pitch = 90° Aligns Two Axes',
        'gimbal_normal': 'Normal (3 DOF)',
        'gimbal_locked': 'Locked (2 DOF)',
        'yaw': 'Yaw', 'pitch': 'Pitch', 'roll': 'Roll',
        'euler_title': 'Euler Angles: Yaw ($\\psi$), Pitch ($\\theta$), Roll ($\\phi$)',
        'axis_angle_title': 'Axis-Angle Rotation',
        'axis': 'Rotation axis $\\hat{n}$',
        'angle': '$\\theta$',
        'quat_interp_title': 'SLERP: Smooth Quaternion Interpolation',
        'start': 'Start', 'end': 'End',
    },
    'zh': {
        'rh_title': '右手坐标系',
        'lh_title': '左手坐标系',
        'ned': 'NED（航空航天）',
        'enu': 'ENU（机器人/ROS）',
        'north': 'N', 'east': 'E', 'down': 'D', 'up': 'U',
        'world': '世界坐标系', 'body': '机体坐标系',
        'world_body_title': '世界坐标系 vs 机体坐标系',
        'active_title': '主动旋转（旋转向量）',
        'passive_title': '被动旋转（变换坐标系）',
        'original': '原始', 'rotated': '旋转后',
        'gimbal_title': '万向锁：俯仰角 = 90° 使两轴重合',
        'gimbal_normal': '正常（3自由度）',
        'gimbal_locked': '锁定（2自由度）',
        'yaw': '偏航', 'pitch': '俯仰', 'roll': '横滚',
        'euler_title': '欧拉角：偏航 ($\\psi$)、俯仰 ($\\theta$)、横滚 ($\\phi$)',
        'axis_angle_title': '轴角旋转',
        'axis': '旋转轴 $\\hat{n}$',
        'angle': '$\\theta$',
        'quat_interp_title': 'SLERP：四元数球面线性插值',
        'start': '起点', 'end': '终点',
    },
}


def _rc(lang):
    base = {
        'font.size': 10,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    }
    if lang == 'zh':
        base['font.family'] = 'sans-serif'
        base['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
        base['axes.unicode_minus'] = False
    else:
        base['font.family'] = 'serif'
    return base


def _draw_frame_3d(ax, R=np.eye(3), origin=np.zeros(3), labels=('x','y','z'),
                   colors=('C3','C2','C0'), length=1.0, lw=2.5, fontsize=12):
    """Draw 3D coordinate axes."""
    for i, (c, lab) in enumerate(zip(colors, labels)):
        d = R[:, i] * length
        ax.quiver(*origin, *d, color=c, arrow_length_ratio=0.12, lw=lw)
        ax.text(*(origin + d * 1.18), lab, color=c, fontsize=fontsize,
                ha='center', va='center', fontweight='bold')


def _setup_3d(ax, elev=25, azim=-60, lim=1.4):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


# ─────────────────────────────────────────────────────────────
# Fig 1: Right-hand vs Left-hand coordinate frames
# ─────────────────────────────────────────────────────────────
def fig_rh_lh(outpath, lang='en'):
    L = S[lang]; plt.rcParams.update(_rc(lang))
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    _draw_frame_3d(ax1, labels=['X','Y','Z'], length=1.2, lw=3.0, fontsize=15)
    ax1.set_title(L['rh_title'], fontsize=14, pad=5)
    _setup_3d(ax1, lim=1.6)

    ax2 = fig.add_subplot(122, projection='3d')
    R_lh = np.diag([1, 1, -1])
    _draw_frame_3d(ax2, R=R_lh, labels=['X','Y','Z'], length=1.2, lw=3.0, fontsize=15)
    ax2.set_title(L['lh_title'], fontsize=14, pad=5)
    _setup_3d(ax2, lim=1.6)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  saved {outpath}')


# ─────────────────────────────────────────────────────────────
# Fig 2: World frame vs Body frame
# ─────────────────────────────────────────────────────────────
def fig_world_body(outpath, lang='en'):
    L = S[lang]; plt.rcParams.update(_rc(lang))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # World frame at origin
    _draw_frame_3d(ax, labels=[r'$x_w$', r'$y_w$', r'$z_w$'],
                   colors=('C3','C2','C0'), length=1.2, lw=2.5)
    ax.text(0, 0, -0.3, L['world'], fontsize=10, ha='center', color='gray')

    # Body frame rotated 30° yaw + 20° pitch, translated
    psi, theta = np.radians(35), np.radians(20)
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi),  np.cos(psi), 0],
                    [0, 0, 1]])
    Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    R_body = Rz @ Ry
    body_origin = np.array([1.5, 1.0, 0.5])
    _draw_frame_3d(ax, R=R_body, origin=body_origin,
                   labels=[r'$x_b$', r'$y_b$', r'$z_b$'],
                   colors=('C1','C4','C9'), length=0.8, lw=2.0)
    ax.text(*(body_origin + np.array([0, 0, -0.35])), L['body'],
            fontsize=10, ha='center', color='C1')

    # Draw a simple robot body (box)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    hw, hh, hd = 0.25, 0.15, 0.4
    corners_body = np.array([
        [-hd,-hw,-hh], [hd,-hw,-hh], [hd,hw,-hh], [-hd,hw,-hh],
        [-hd,-hw, hh], [hd,-hw, hh], [hd,hw, hh], [-hd,hw, hh],
    ])
    corners_world = (R_body @ corners_body.T).T + body_origin
    faces = [[corners_world[j] for j in f] for f in
             [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.15, color='C1',
                                          edgecolor='C1', linewidth=0.5))

    ax.set_title(L['world_body_title'], fontsize=12)
    _setup_3d(ax, elev=20, azim=-50, lim=2.5)
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  saved {outpath}')


# ─────────────────────────────────────────────────────────────
# Fig 3: Active vs Passive rotation (2D for clarity)
# ─────────────────────────────────────────────────────────────
def fig_active_passive(outpath, lang='en'):
    L = S[lang]; plt.rcParams.update(_rc(lang))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    theta = np.radians(40)
    ct, st = np.cos(theta), np.sin(theta)

    # -- Active: rotate the vector --
    ax1.set_aspect('equal')
    ax1.arrow(0, 0, 0.9, 0, head_width=0.05, head_length=0.05, fc='C0', ec='C0', lw=1.5)
    ax1.arrow(0, 0, 0.9*ct, 0.9*st, head_width=0.05, head_length=0.05, fc='C3', ec='C3', lw=1.5)
    # Arc
    arc_t = np.linspace(0, theta, 30)
    ax1.plot(0.35*np.cos(arc_t), 0.35*np.sin(arc_t), 'k', lw=0.8)
    ax1.text(0.42, 0.12, r'$\theta$', fontsize=11)
    ax1.text(0.95, -0.1, r'$\mathbf{v}$', fontsize=12, color='C0')
    ax1.text(0.9*ct+0.05, 0.9*st+0.05, r"$\mathbf{v}'=R\mathbf{v}$", fontsize=11, color='C3')
    # Axes
    ax1.axhline(0, color='gray', lw=0.3); ax1.axvline(0, color='gray', lw=0.3)
    ax1.set_xlim(-0.3, 1.3); ax1.set_ylim(-0.3, 1.1)
    ax1.set_title(L['active_title'], fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([])

    # -- Passive: rotate the frame --
    ax2.set_aspect('equal')
    # Original frame
    ax2.arrow(0, 0, 0.8, 0, head_width=0.04, head_length=0.04, fc='gray', ec='gray', lw=1, alpha=0.5)
    ax2.arrow(0, 0, 0, 0.8, head_width=0.04, head_length=0.04, fc='gray', ec='gray', lw=1, alpha=0.5)
    ax2.text(0.85, -0.08, r'$x$', color='gray', fontsize=10)
    ax2.text(-0.1, 0.85, r'$y$', color='gray', fontsize=10)
    # Rotated frame
    ax2.arrow(0, 0, 0.8*ct, 0.8*st, head_width=0.04, head_length=0.04, fc='C2', ec='C2', lw=1.5)
    ax2.arrow(0, 0, -0.8*st, 0.8*ct, head_width=0.04, head_length=0.04, fc='C4', ec='C4', lw=1.5)
    ax2.text(0.8*ct+0.05, 0.8*st+0.02, r"$x'$", color='C2', fontsize=10)
    ax2.text(-0.8*st-0.12, 0.8*ct+0.02, r"$y'$", color='C4', fontsize=10)
    # Vector (same in space)
    vx, vy = 0.7, 0.5
    ax2.arrow(0, 0, vx, vy, head_width=0.04, head_length=0.04, fc='C3', ec='C3', lw=1.5)
    ax2.text(vx+0.05, vy+0.02, r'$\mathbf{v}$', fontsize=12, color='C3')
    ax2.axhline(0, color='gray', lw=0.3); ax2.axvline(0, color='gray', lw=0.3)
    ax2.set_xlim(-0.5, 1.1); ax2.set_ylim(-0.3, 1.1)
    ax2.set_title(L['passive_title'], fontsize=11)
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  saved {outpath}')


# ─────────────────────────────────────────────────────────────
# Fig 4: Euler angles (yaw-pitch-roll) visualization
# ─────────────────────────────────────────────────────────────
def fig_euler_angles(outpath, lang='en'):
    L = S[lang]; plt.rcParams.update(_rc(lang))
    fig = plt.figure(figsize=(11, 4.5))

    angles_list = [
        (30, 0, 0, f'{L["yaw"]} $\\psi=30°$'),
        (30, 20, 0, f'+ {L["pitch"]} $\\theta=20°$'),
        (30, 20, 15, f'+ {L["roll"]} $\\phi=15°$'),
    ]

    for idx, (psi_d, theta_d, phi_d, title) in enumerate(angles_list):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        psi, theta, phi = np.radians(psi_d), np.radians(theta_d), np.radians(phi_d)

        Rz = np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        Rx = np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
        R = Rz @ Ry @ Rx

        # World frame (faint)
        _draw_frame_3d(ax, labels=['x','y','z'], colors=('lightcoral','lightgreen','lightblue'),
                       length=0.7, lw=1.0, fontsize=9)
        # Body frame
        _draw_frame_3d(ax, R=R, labels=[r"$x_b$",r"$y_b$",r"$z_b$"],
                       colors=('C3','C2','C0'), length=1.0, lw=2.0, fontsize=10)

        ax.set_title(title, fontsize=10, pad=0)
        _setup_3d(ax, elev=25, azim=-55, lim=1.5)

    fig.savefig(outpath)
    plt.close(fig)
    print(f'  saved {outpath}')


# ─────────────────────────────────────────────────────────────
# Fig 5: Gimbal lock visualization
# ─────────────────────────────────────────────────────────────
def fig_gimbal_lock(outpath, lang='en'):
    L = S[lang]; plt.rcParams.update(_rc(lang))
    fig = plt.figure(figsize=(10, 5))

    # Normal case: pitch = 20°
    ax1 = fig.add_subplot(121, projection='3d')
    psi, theta, phi = np.radians(25), np.radians(20), np.radians(15)
    Rz = np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
    Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    Rx = np.array([[1,0,0],[0,np.cos(phi),-np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])

    # Draw three gimbal rings
    t = np.linspace(0, 2*np.pi, 100)
    ring = np.array([np.cos(t), np.sin(t), np.zeros_like(t)])

    # Yaw ring (world z)
    r_yaw = ring * 1.2
    ax1.plot(*r_yaw, 'C0', lw=1.5, alpha=0.6)
    # Pitch ring
    r_pitch = (Rz @ (ring * 1.0))
    ax1.plot(*r_pitch, 'C2', lw=1.5, alpha=0.6)
    # Roll ring
    r_roll = (Rz @ Ry @ np.array([np.zeros_like(t), np.cos(t), np.sin(t)])) * 0.8
    ax1.plot(*r_roll, 'C3', lw=1.5, alpha=0.6)

    _draw_frame_3d(ax1, R=Rz@Ry@Rx, labels=[r'$x_b$',r'$y_b$',r'$z_b$'],
                   colors=('C3','C2','C0'), length=0.7, lw=2)
    ax1.set_title(L['gimbal_normal'], fontsize=11, pad=0)
    _setup_3d(ax1, lim=1.6)

    # Locked case: pitch = 90°
    ax2 = fig.add_subplot(122, projection='3d')
    theta2 = np.radians(90)
    Ry2 = np.array([[np.cos(theta2),0,np.sin(theta2)],[0,1,0],[-np.sin(theta2),0,np.cos(theta2)]])

    r_yaw2 = ring * 1.2
    ax2.plot(*r_yaw2, 'C0', lw=1.5, alpha=0.6)
    r_pitch2 = (Rz @ (ring * 1.0))
    ax2.plot(*r_pitch2, 'C2', lw=1.5, alpha=0.6)
    # Roll ring now aligned with yaw!
    r_roll2 = (Rz @ Ry2 @ np.array([np.zeros_like(t), np.cos(t), np.sin(t)])) * 0.8
    ax2.plot(*r_roll2, 'C3', lw=2.5, ls='--', alpha=0.8)

    _draw_frame_3d(ax2, R=Rz@Ry2, labels=[r'$x_b$',r'$y_b$',r'$z_b$'],
                   colors=('C3','C2','C0'), length=0.7, lw=2)
    ax2.set_title(L['gimbal_locked'], fontsize=11, pad=0, color='C3')
    _setup_3d(ax2, lim=1.6)

    fig.suptitle(L['gimbal_title'], fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  saved {outpath}')


# ─────────────────────────────────────────────────────────────
# Fig 6: Axis-angle rotation
# ─────────────────────────────────────────────────────────────
def fig_axis_angle(outpath, lang='en'):
    L = S[lang]; plt.rcParams.update(_rc(lang))
    fig = plt.figure(figsize=(6, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    # Rotation axis
    n = np.array([0.3, 0.2, 1.0])
    n = n / np.linalg.norm(n)
    ax.quiver(0, 0, 0, *(n*1.3), color='k', arrow_length_ratio=0.08, lw=2)
    ax.text(*(n*1.45), L['axis'], fontsize=10, ha='center')

    # Original vector
    v = np.array([1.0, 0.3, 0.2])
    v = v / np.linalg.norm(v) * 0.9
    ax.quiver(0, 0, 0, *v, color='C0', arrow_length_ratio=0.1, lw=2)
    ax.text(*(v*1.15), r'$\mathbf{v}$', fontsize=12, color='C0')

    # Rotated vector (Rodrigues)
    theta_rot = np.radians(60)
    v_rot = (v * np.cos(theta_rot)
             + np.cross(n, v) * np.sin(theta_rot)
             + n * np.dot(n, v) * (1 - np.cos(theta_rot)))
    ax.quiver(0, 0, 0, *v_rot, color='C3', arrow_length_ratio=0.1, lw=2)
    ax.text(*(v_rot*1.15), r"$\mathbf{v}'$", fontsize=12, color='C3')

    # Draw arc from v to v'
    # Project v onto plane perpendicular to n
    v_par = np.dot(v, n) * n
    v_perp = v - v_par
    v_perp2 = np.cross(n, v_perp)
    r_arc = np.linalg.norm(v_perp)
    arc_angles = np.linspace(0, theta_rot, 40)
    arc_pts = np.array([v_par + r_arc*(np.cos(a)*v_perp/r_arc + np.sin(a)*v_perp2/r_arc)
                        for a in arc_angles])
    ax.plot(arc_pts[:,0], arc_pts[:,1], arc_pts[:,2], 'C3--', lw=1, alpha=0.7)

    # Angle label
    mid_idx = len(arc_angles)//2
    ax.text(*(arc_pts[mid_idx]*1.15), L['angle'], fontsize=12, color='C3')

    # Dashed projection line
    ax.plot([v[0], v_par[0]], [v[1], v_par[1]], [v[2], v_par[2]], 'k:', lw=0.8, alpha=0.5)
    ax.plot([v_rot[0], v_par[0]], [v_rot[1], v_par[1]], [v_rot[2], v_par[2]], 'k:', lw=0.8, alpha=0.5)

    ax.set_title(L['axis_angle_title'], fontsize=12)
    _setup_3d(ax, elev=20, azim=-45, lim=1.4)
    fig.savefig(outpath)
    plt.close(fig)
    print(f'  saved {outpath}')


# ── Main ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating rotation chapter figures...')
    for lang, fdir in [('en', 'figures'), ('zh', 'figures_zh')]:
        fig_rh_lh(f'{fdir}/coord_rh_lh.pdf', lang)
        fig_world_body(f'{fdir}/coord_world_body.pdf', lang)
        fig_active_passive(f'{fdir}/rotation_active_passive.pdf', lang)
        fig_euler_angles(f'{fdir}/euler_angles.pdf', lang)
        fig_gimbal_lock(f'{fdir}/gimbal_lock.pdf', lang)
        fig_axis_angle(f'{fdir}/axis_angle.pdf', lang)
    print('Done.')

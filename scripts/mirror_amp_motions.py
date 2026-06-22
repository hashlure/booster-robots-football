#!/usr/bin/env python3
"""AMP 动作数据左右镜像 — 生成镜像版 .txt 文件。

支持两种 56 维格式（自动检测 K1=22 关节 / T1=23 关节）：
  Format A (AMP training / gmr_data_conversion):
    [0:3] root_pos, [3:6] euler,
    [6:6+J] dof_pos,
    [6+J:6+J+3] lin_vel, [6+J+3:6+J+6] ang_vel,
    [6+J+6:6+J+6+J] dof_vel
  Format B (replay_amp_txt --save_path):
    [0:J] dof_pos, [J:2J] dof_vel,
    [2J:2J+3] left_hand, [2J+3:2J+6] right_hand,
    [2J+6:2J+9] left_foot, [2J+9:2J+12] right_foot

用法: python mirror_amp_motions.py <input.txt> [output.txt]
"""

import json
import sys
import os
import copy
import argparse

# K1 关节镜像映射 (GMR/AMP 顺序, 22 关节)
# 0:AAHead_yaw, 1:Head_pitch, 2:L_Shoulder_Pitch, 3:L_Shoulder_Roll, 4:L_Elbow_Pitch, 5:L_Elbow_Yaw,
# 6:R_Shoulder_Pitch, 7:R_Shoulder_Roll, 8:R_Elbow_Pitch, 9:R_Elbow_Yaw,
# 10:L_Hip_Pitch, 11:L_Hip_Roll, 12:L_Hip_Yaw, 13:L_Knee_Pitch, 14:L_Ankle_Pitch, 15:L_Ankle_Roll,
# 16:R_Hip_Pitch, 17:R_Hip_Roll, 18:R_Hip_Yaw, 19:R_Knee_Pitch, 20:R_Ankle_Pitch, 21:R_Ankle_Roll
JOINT_MIRROR_MAP_22 = [
    (0,  0,  -1), (1,  1,   1),
    (2,  6,   1), (3,  7,  -1), (4,  8,   1), (5,  9,  -1),
    (6,  2,   1), (7,  3,  -1), (8,  4,   1), (9,  5,  -1),
    (10, 16,  1), (11, 17, -1), (12, 18, -1), (13, 19,  1), (14, 20,  1), (15, 21, -1),
    (16, 10,  1), (17, 11, -1), (18, 12, -1), (19, 13,  1), (20, 14,  1), (21, 15, -1),
]

# T1 关节镜像映射 (23 关节 — 比 K1 多一个 Waist)
JOINT_MIRROR_MAP_23 = [
    (0,  0,  -1), (1,  1,   1),
    (2,  7,   1), (3,  8,  -1), (4,  9,   1), (5,  10, -1),
    (6,  6,   1),   # Waist → keep
    (7,  2,   1), (8,  3,  -1), (9,  4,   1), (10, 5,  -1),
    (11, 17,  1), (12, 18, -1), (13, 19, -1), (14, 20,  1), (15, 21,  1), (16, 22, -1),
    (17, 11,  1), (18, 12, -1), (19, 13, -1), (20, 14,  1), (21, 15,  1), (22, 16, -1),
]

def get_pos_mirror_map(J):
    """Format B 3D position mirror map — dynamic based on joint count.
    Layout: dof_pos(J) + dof_vel(J) + LH(3) + RH(3) + LF(3) + RF(3)"""
    b = 2 * J  # hand/foot start offset
    return [
        (b,     b+3, -1),   # left_hand  → right_hand  (y 取反)
        (b+3,   b,   -1),   # right_hand → left_hand
        (b+6,   b+9, -1),   # left_foot  → right_foot
        (b+9,   b+6, -1),   # right_foot → left_foot
    ]


def detect_format(motion_data, n_dims):
    """Auto-detect format A (AMP training) or B (replay save)."""
    n_joints = None
    if n_dims == 56:
        n_joints = 22
    elif n_dims == 58:
        n_joints = 23
    elif n_dims == 68:  # 23 joints * 2 + 22 hand/foot
        n_joints = 23

    # Heuristic: format A has small values at [0:3] (root_pos in meters)
    # format B has joint angles (rad) at [0:3]
    first3 = motion_data["Frames"][0][:3]
    # Format A: root_pos ~ [-2, 2] (meters); Format B: dof_pos ~ [-3, 3] (rad)
    # Not always distinguishable, so default to format A for 56/58 dims
    if n_joints is not None:
        return "A", n_joints
    return None, None


def mirror_frame_format_a(frame, n_joints):
    """镜像 Format A: root_pos + euler + dof_pos + lin_vel + ang_vel + dof_vel."""
    m = [0.0] * len(frame)
    J = n_joints
    joint_map = JOINT_MIRROR_MAP_22 if J == 22 else JOINT_MIRROR_MAP_23

    # root_pos [0:3]: x→keep, y→-y, z→keep
    m[0], m[1], m[2] = frame[0], -frame[1], frame[2]
    # euler [3:6]: roll→-roll, pitch→keep, yaw→-yaw
    m[3], m[4], m[5] = -frame[3], frame[4], -frame[5]
    # dof_pos [6:6+J]
    for s, t, sign in joint_map:
        m[6 + t] = sign * frame[6 + s]
    # lin_vel [6+J : 6+J+3]: same as root_pos
    b = 6 + J
    m[b], m[b+1], m[b+2] = frame[b], -frame[b+1], frame[b+2]
    # ang_vel [b+3 : b+6]: same as euler
    m[b+3], m[b+4], m[b+5] = -frame[b+3], frame[b+4], -frame[b+5]
    # dof_vel [b+6 : b+6+J]
    for s, t, sign in joint_map:
        m[b + 6 + t] = sign * frame[b + 6 + s]
    return m


def mirror_frame_format_b(frame, n_joints):
    """镜像 Format B: dof_pos + dof_vel + hand/foot positions."""
    m = [0.0] * len(frame)
    J = n_joints
    joint_map = JOINT_MIRROR_MAP_22 if J == 22 else JOINT_MIRROR_MAP_23

    # dof_pos [0:J]
    for s, t, sign in joint_map:
        m[t] = sign * frame[s]
    # dof_vel [J:2J]
    for s, t, sign in joint_map:
        m[J + t] = sign * frame[J + s]
    # 3D positions (hand/foot): swap left⇄right, y inverts
    pos_map = get_pos_mirror_map(J)
    for src_start, tgt_start, y_sign in pos_map:
        x, y, z = frame[src_start], frame[src_start + 1], frame[src_start + 2]
        m[tgt_start] = x
        m[tgt_start + 1] = y_sign * y
        m[tgt_start + 2] = z
    return m


def mirror_file(src_path, dst_path, fmt=None):
    with open(src_path) as f:
        data = json.load(f)

    frames = data["Frames"]
    n_dims = len(frames[0])

    if fmt is None:
        fmt, n_joints = detect_format(data, n_dims)
        if fmt is None:
            raise ValueError(f"Cannot detect format for {n_dims}-dim frames")
    else:
        n_joints = 22 if n_dims == 56 else 23

    print(f"  Format {fmt}, {n_joints} joints, {n_dims} dims, {len(frames)} frames")

    mirror_fn = mirror_frame_format_a if fmt == "A" else mirror_frame_format_b
    mirrored = copy.deepcopy(data)
    mirrored["Frames"] = [mirror_fn(f, n_joints) for f in frames]

    with open(dst_path, 'w') as f:
        json.dump(mirrored, f, indent=2)

    return len(mirrored["Frames"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .txt file")
    parser.add_argument("output", nargs="?", help="Output .txt file (default: xxx_mirror.txt)")
    parser.add_argument("--format", choices=["A", "B"], default=None, help="Force format (A=AMP training, B=replay save)")
    args = parser.parse_args()

    src = args.input
    dst = args.output
    if dst is None:
        name, ext = os.path.splitext(src)
        dst = f"{name}_mirror{ext}"

    n = mirror_file(src, dst, args.format)
    print(f"  Done: {src} → {dst}  ({n} frames)")

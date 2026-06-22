#!/usr/bin/env python3
"""Convert GMR pkl to AMP txt — Format B (dof_pos + dof_vel + hand/foot FK).

Format B layout:
  [0:J]      dof_pos        (J joint angles)
  [J:2J]     dof_vel        (J joint velocities)
  [2J:2J+3]  left_hand_pos  (robot-local FK, < 1m)
  [2J+3:2J+6] right_hand_pos
  [2J+6:2J+9] left_foot_pos
  [2J+9:2J+12] right_foot_pos
  Total: 2*J + 12 = 58 (J=23, T1) or 56 (J=22, K1)
"""

import pickle
import numpy as np
import argparse


# GMR → Isaac Lab 关节重排映射
# 来源: replay_amp_txt.py reorder() 函数
GMR_TO_ISAAC = {
    # T1: 23 关节
    23: [0, 2, 6, 10, 1, 3, 7, 11, 17, 4, 8, 12, 18, 5, 9, 13, 19, 14, 20, 15, 21, 16, 22],
    # K1: 22 关节
    22: [0, 2, 6, 10, 16, 1, 3, 7, 11, 17, 4, 8, 12, 18, 5, 9, 13, 19, 14, 20, 15, 21],
}


def convert_pkl_to_format_b(input_pkl, output_txt, fps=30.0):
    dt = 1.0 / fps

    with open(input_pkl, "rb") as f:
        pkl = pickle.load(f)

    dof_pos = pkl["dof_pos"]          # (N, J) — GMR order
    N, J = dof_pos.shape

    # Reorder joints GMR → Isaac Lab
    if J in GMR_TO_ISAAC:
        dof_pos = dof_pos[:, GMR_TO_ISAAC[J]]
    local_body_pos = pkl["local_body_pos"]  # (N, B, 3)
    body_names = pkl["link_body_list"]      # [B]

    # Find hand/foot body indices
    body_lookup = {name: i for i, name in enumerate(body_names)}
    try:
        lh = body_lookup["left_hand_link"]
        rh = body_lookup["right_hand_link"]
        lf = body_lookup["left_foot_link"]
        rf = body_lookup["right_foot_link"]
    except KeyError as e:
        raise KeyError(f"Body name {e} not found in link_body_list: {body_names}")

    # Joint velocities (N-1 frames → drop last frame of positions to align)
    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt  # (N-1, J)

    # FK hand/foot positions (robot-local frame, already computed in pkl)
    left_hand = local_body_pos[:-1, lh, :]   # (N-1, 3)
    right_hand = local_body_pos[:-1, rh, :]
    left_foot = local_body_pos[:-1, lf, :]
    right_foot = local_body_pos[:-1, rf, :]

    # Build Format B: dof_pos + dof_vel + L_hand + R_hand + L_foot + R_foot
    data_output = np.concatenate(
        (dof_pos[:-1], dof_vel, left_hand, right_hand, left_foot, right_foot),
        axis=1,
    )
    # (N-1) frames, 2*J + 12 dims

    # Write JSON-like txt
    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames = f.readlines()

    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')
        for i, line in enumerate(frames):
            end = ']\n' if i == len(frames) - 1 else '],\n'
            f.write('  [' + line.rstrip() + end)
        f.write(']\n}')

    # Verify hand/foot positions are reasonable
    hf = np.concatenate([left_hand, right_hand, left_foot, right_foot], axis=1)
    hf_max = np.abs(hf).max()
    if hf_max > 5.0:
        print(f"  ⚠️  WARNING: max hand/foot pos = {hf_max:.1f}m (expected < 2m in robot frame)")

    print(f"Done: {input_pkl} -> {output_txt}  ({len(frames)} frames, {data_output.shape[1]} dims, hand/foot max={hf_max:.2f}m)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    convert_pkl_to_format_b(args.input_pkl, args.output_txt, args.fps)

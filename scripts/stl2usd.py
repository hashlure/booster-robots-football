#!/usr/bin/env python3
"""Parse binary STL and write USD (.usda) mesh file. No dependencies beyond Python stdlib."""

import struct
import os
import sys

def parse_stl_binary(path):
    """Parse binary STL file, return (vertices, faces)."""
    with open(path, 'rb') as f:
        f.read(80)  # header
        n_tri = struct.unpack('<I', f.read(4))[0]
        verts = []
        faces = []
        seen = {}
        for i in range(n_tri):
            f.read(12)  # normal
            tri_verts = []
            for _ in range(3):
                v = struct.unpack('<fff', f.read(12))
                key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                if key not in seen:
                    seen[key] = len(verts)
                    verts.append(v)
                tri_verts.append(seen[key])
            faces.append(tuple(tri_verts))
            f.read(2)  # attribute
    return verts, faces


def write_usda(output_path, verts, faces, scale=1.0):
    """Write USD Ascii file with mesh data, centered at origin."""
    # Compute center
    cx = (min(v[0] for v in verts) + max(v[0] for v in verts)) / 2
    cy = (min(v[1] for v in verts) + max(v[1] for v in verts)) / 2
    cz = (min(v[2] for v in verts) + max(v[2] for v in verts)) / 2
    print(f"  Center: ({cx*scale:.3f}, {cy*scale:.3f}, {cz*scale:.3f})")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('#usda 1.0\n')
        f.write('(\n')
        f.write('    defaultPrim = "goal_door"\n')
        f.write('    metersPerUnit = 1.0\n')
        f.write('    upAxis = "Z"\n')
        f.write(')\n\n')

        f.write('def Xform "goal_door" (\n')
        f.write('    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI", "PhysicsCollisionAPI"]\n')
        f.write(')\n')
        f.write('{\n')
        f.write('    bool physics:kinematicEnabled = true\n')
        f.write('    bool physics:rigidBodyEnabled = true\n\n')

        f.write('    def Mesh "goal_mesh" (\n')
        f.write('        prepend apiSchemas = ["PhysicsCollisionAPI"]\n')
        f.write('    )\n')
        f.write('    {\n')
        f.write(f'        point3f[] points = [\n')
        for i, (x, y, z) in enumerate(verts):
            comma = ',' if i < len(verts) - 1 else ''
            sx, sy, sz = (x - cx) * scale, (y - cy) * scale, (z - cz) * scale
            f.write(f'            ({sx:.6f}, {sy:.6f}, {sz:.6f}){comma}\n')
        f.write('        ]\n\n')

        f.write('        int[] faceVertexCounts = [\n')
        for i in range(len(faces)):
            comma = ',' if i < len(faces) - 1 else ''
            f.write(f'            3{comma}\n')
        f.write('        ]\n\n')

        f.write('        int[] faceVertexIndices = [\n')
        flat = [str(idx) for face in faces for idx in face]
        chunk = 30
        for i in range(0, len(flat), chunk):
            segment = flat[i:i+chunk]
            end = ',\n' if i + chunk < len(flat) else '\n'
            f.write(f'            {", ".join(segment)}{end}')
        f.write('        ]\n')

        f.write('    }\n')
        f.write('}\n')

    size_kb = os.stat(output_path).st_size / 1024
    print(f"Done: {output_path}  ({len(verts)} verts, {len(faces)} faces, {size_kb:.0f} KB)")


if __name__ == '__main__':
    stl = sys.argv[1] if len(sys.argv) > 1 else '/root/booster_amp_lab/goal_door.STL'
    usd = sys.argv[2] if len(sys.argv) > 2 else '/root/booster_amp_lab/booster_assets/models/goal_door.usda'
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    v, f = parse_stl_binary(stl)
    write_usda(usd, v, f, scale)

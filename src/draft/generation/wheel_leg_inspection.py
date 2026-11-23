"""Geometric sitting inspection for the documented Galileo X serial chain.

This is an explicit inspection tool, not a controller or a generator gate for
unrelated robots. Distances bypass MuJoCo's parent/grandparent contact filters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import mujoco
import numpy as np


TOLERANCE = 2e-6  # six-significant-figure MJCF round-trip


@dataclass
class SittingInspection:
    report: dict
    fractions: np.ndarray
    qpos: np.ndarray


def validate_wheel_dimensions(params: dict) -> None:
    for name in ('wheel_radius', 'wheel_width', 'wheel_mass', 'assembly_clearance',
                 'sitting_lift_height'):
        value = float(params[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name}: expected a positive finite value')
    clearance = float(params['assembly_clearance'])
    if params['wheel_radius'] < params['j4_motor_r'] + clearance - TOLERANCE:
        raise ValueError('wheel_radius: wheel does not clear the J4 motor envelope')
    if params['wheel_width'] < params['j4_motor_L'] + 2 * clearance - TOLERANCE:
        raise ValueError('wheel_width: wheel does not contain the J4 motor envelope and clearance')


def sitting_path(model: mujoco.MjModel, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Level torso, contact FK first, then a smooth extra fold in the last 50 mm."""
    validate_wheel_dimensions(params)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'standing')
    if key < 0:
        raise ValueError('sitting inspection needs a standing keyframe')
    start = model.key_qpos[key].copy()
    legs = [leg['name'] for leg in params['_legs']]
    root = model.joint('torso_link_freejoint').qposadr[0]
    z0, ze = float(start[root + 2]), float(params['torso_lz']) / 2
    if z0 <= ze:
        raise ValueError('sitting path: standing torso is already at or below the floor')
    joints = {leg: [model.joint(f'{leg}_j{i}').id for i in (1, 2, 3, 4)] for leg in legs}
    jfirst = joints[legs[0]]
    a0, b0 = (float(start[model.jnt_qposadr[jfirst[i]]]) for i in (0, 2))
    ae, be = (math.radians(float(params[k])) for k in ('sitting_j1_deg', 'sitting_j3_deg'))
    if not all(math.isfinite(v) for v in (ae, be)):
        raise ValueError('sitting angles must be finite')
    cant = float(params['thigh_cant_rad'])
    upper, lower = (float(params[k]) for k in ('upper_leg_link_length', 'lower_leg_link_length'))
    offset, radius = float(params['j2_output_offset']), float(params['wheel_radius'])

    def contact_angle(height, knee):
        A = offset + upper * math.cos(cant) + lower * math.cos(cant + knee)
        B = upper * math.sin(cant) + lower * math.sin(cant + knee)
        norm = math.hypot(A, B)
        ratio = (height - radius) / norm if norm else math.inf
        if not math.isfinite(ratio) or abs(ratio) > 1 + 1e-9:
            raise ValueError(f'sitting path: contact FK has no solution at height {height:.6g} m')
        return math.atan2(B, A) - math.acos(max(-1.0, min(1.0, ratio)))

    if abs(contact_angle(z0, b0) - a0) > 1e-4:
        raise ValueError('sitting path: standing pose is not on the supported contact-FK branch')
    extra_fold = contact_angle(ze, be) - ae
    if extra_fold < 0:
        raise ValueError('sitting_j1_deg: endpoint would push wheels through the floor')

    def pose(t):
        z, knee = z0 + t * (ze - z0), b0 + t * (be - b0)
        u = min(1.0, max(0.0, (ze + params['sitting_lift_height'] - z)
                             / params['sitting_lift_height']))
        hip = contact_angle(z, knee) - u * u * (3 - 2 * u) * extra_fold
        qpos = start.copy()
        qpos[root:root + 7] = [0, 0, z, 1, 0, 0, 0]
        for leg in legs:
            for jid, value in zip(joints[leg], (hip, 0, knee, 0)):
                if model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    if not lo - TOLERANCE <= value <= hi + TOLERANCE:
                        raise ValueError(f'sitting path: {model.joint(jid).name} position '
                                         f'{value:.6g} is outside range [{lo:.6g}, {hi:.6g}]')
                qpos[model.jnt_qposadr[jid]] = value
        return qpos

    count = max(2, math.ceil((z0 - ze) / 0.001), math.ceil(abs(be - b0) / math.radians(0.5)))
    times = np.linspace(0, 1, count + 1)
    for _ in range(20):
        poses = np.array([pose(float(t)) for t in times])
        adr = [model.jnt_qposadr[jid] for ids in joints.values() for jid in ids]
        angle_steps = np.max(np.abs(np.diff(poses[:, adr], axis=0)), axis=1)
        bad = angle_steps > math.radians(0.5) + 1e-12
        if not np.any(bad):
            return times, poses
        times = np.sort(np.concatenate([times, (times[:-1][bad] + times[1:][bad]) / 2]))
    raise ValueError('sitting path sampling did not converge')


def floor_gap(model: mujoco.MjModel, data: mujoco.MjData, gi: int) -> float:
    """Exact support height for the primitives/convex torso used by this model."""
    kind, size = model.geom_type[gi], model.geom_size[gi]
    rotation = data.geom_xmat[gi].reshape(3, 3)
    z = float(data.geom_xpos[gi, 2])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        nz = float(rotation[2, 2])
        return z - size[0] * math.sqrt(max(0.0, 1 - nz * nz)) - size[1] * abs(nz)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return z - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return z - size[0] - size[1] * abs(float(rotation[2, 2]))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return z - float(np.abs(rotation[2]) @ size)
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gi]
        lo, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[lo:lo + count]
        return z + float(np.min(verts @ rotation[2]))
    raise ValueError(f'unsupported floor-distance geom: {model.geom(gi).name}')


def _bolted_pairs(legs: list[str]) -> dict[frozenset, str]:
    # Explicit physical seams only. In particular, wheel/thigh, wheel/shank and
    # shank/J3 pairs remain live even if the generated contact filters hide them.
    seams = [
        ('j1_housing_motor', 'j1_link_link', 'J1 output flange'),
        ('j1_link_motor', 'j2_link_link', 'J2 output flange'),
        ('j2_link_link', 'thigh_link_link', 'fixed canted mount'),
        ('thigh_link_motor', 'j3_output_link', 'J3 output flange'),
        ('j3_output_link', 'shank_link_link', 'rigid outboard corner'),
        ('shank_link_link', 'j4_housing_link', 'rigid distal corner'),
        ('j4_housing_link', 'wheel_geom', 'stationary shaft inside wheel hub proxy'),
        ('j4_housing_motor', 'wheel_geom', 'motor nested inside wheel hub proxy'),
    ]
    return {frozenset((f'{leg}_{a}', f'{leg}_{b}')): why
            for leg in legs for a, b, why in seams}


def inspect_sitting(model: mujoco.MjModel, params: dict,
                    *, raise_on_error: bool = True) -> SittingInspection:
    fractions, poses = sitting_path(model, params)
    data = mujoco.MjData(model)
    legs = [leg['name'] for leg in params['_legs']]
    seams = _bolted_pairs(legs)
    solids = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    solids = [i for i in solids if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE]
    pairs = [(a, b) for a, b in combinations(solids, 2)
             if model.geom_bodyid[a] != model.geom_bodyid[b]
             and frozenset((model.geom(a).name, model.geom(b).name)) not in seams]
    wheels = [model.geom(f'{leg}_wheel_geom').id for leg in legs]
    torso = model.geom('torso_link_mesh').id
    minimum = {'parts': None, 'hardware_floor': None, 'wheel_floor': None, 'torso_floor': None}
    errors: list[str] = []

    def record(kind, gap, names, index):
        current = minimum[kind]
        if current is None or gap < current['distance_m']:
            minimum[kind] = {'distance_m': float(gap), 'geoms': names,
                             'sample': index, 'fraction': float(fractions[index]),
                             'qpos': poses[index].tolist()}

    for index, qpos in enumerate(poses):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for a, b in pairs:
            distance = mujoco.mj_geomDistance(model, data, a, b, 0.1, None)
            record('parts', distance, [model.geom(a).name, model.geom(b).name], index)
        for gi in solids:
            gap = floor_gap(model, data, gi)
            kind = 'torso_floor' if gi == torso else 'wheel_floor' if gi in wheels else 'hardware_floor'
            record(kind, gap, [model.geom(gi).name, 'floor'], index)
    clearance = float(params['assembly_clearance'])
    for kind, value in minimum.items():
        required = clearance if kind == 'parts' else 0.0
        if value is not None and value['distance_m'] < required - TOLERANCE:
            errors.append(f"{kind}: {' / '.join(value['geoms'])} distance "
                          f"{value['distance_m'] * 1000:.3f} mm at sample {value['sample']} "
                          f"is below {required * 1000:.3f} mm")
    adr = [model.jnt_qposadr[model.joint(f'{leg}_j{i}').id] for leg in legs for i in (1, 2, 3, 4)]
    report = {
        'valid': not errors, 'kind': 'geometry_only',
        'limitations': 'Sampled geometric path with late wheel lift; no stability, controller or load validation.',
        'samples': len(poses), 'tolerance_m': TOLERANCE,
        'max_joint_step_deg': float(np.degrees(np.max(np.abs(np.diff(poses[:, adr], axis=0))))),
        'max_base_step_m': float(np.max(np.abs(np.diff(poses[:, 2])))),
        'minimum': minimum, 'errors': errors,
        'intentional_seams': [{'geoms': sorted(pair), 'reason': reason} for pair, reason in seams.items()],
        'end_wheel_floor_gaps_m': {leg: floor_gap(model, data, gi) for leg, gi in zip(legs, wheels)},
        'end_torso_floor_gap_m': floor_gap(model, data, torso),
        'end_joint_degrees': {f'{leg}_j{i}': math.degrees(float(poses[-1, model.jnt_qposadr[jid]]))
                              for leg in legs for i in (1, 2, 3, 4)
                              for jid in [model.joint(f'{leg}_j{i}').id]},
    }
    if errors and raise_on_error:
        raise ValueError('sitting inspection failed: ' + '; '.join(errors))
    return SittingInspection(report, fractions, poses)

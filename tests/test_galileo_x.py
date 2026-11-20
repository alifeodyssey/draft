"""Spatial FK, symmetry, assembly and editor integration for the concept robot."""

import asyncio
import math
import threading
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from draft.generation.generator import RobotGenerator
from draft.paths import repo_root
from draft.robots import robot_dir
from draft.trends import FeasibilityViolation


LEGS = [('fl', 1, 1), ('fr', 1, -1), ('rl', -1, 1), ('rr', -1, -1)]


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp('galileo_x')
    RobotGenerator(robot_dir('galileo_x')).generate(out)
    model = mujoco.MjModel.from_xml_path(str(out / 'galileo_x.xml'))
    params = yaml.safe_load((out / 'parameters_resolved.yaml').read_text(encoding='utf-8'))
    report = yaml.safe_load((out / 'feasibility_report.yaml').read_text(encoding='utf-8'))
    return model, params, report


def _pose(model, angles):
    data = mujoco.MjData(model)
    for leg, q in angles.items():
        for index, value in enumerate(q, 1):
            joint = model.joint(f'{leg}_j{index}')
            data.qpos[joint.qposadr[0]] = value
    mujoco.mj_forward(model, data)
    return data


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _analytical_positions(params, sx, sy, q):
    """Independent serial-chain equations, rather than reading the resolved tree."""
    j1 = np.array([sx * params['j1_mount_x'],
                   sy * params['j1_mount_y'], params['standing_height']])
    hip = _ry(sx * q[0])
    j2 = j1 + hip @ [0, sy * params['hip_abduction_link_length'], 0]
    yaw = hip @ _rz(sx * sy * q[1])
    mount = j2 + yaw @ [0, 0, -params['j2_output_offset']]
    thigh = yaw @ _ry(-sx * params['thigh_cant_rad'])
    knee = mount + thigh @ [0, 0, -params['upper_leg_link_length']]
    foot = knee + thigh @ _ry(-sx * q[2]) @ [0, 0, -params['lower_leg_link_length']]
    return j1, j2, knee, foot


def _assert_end_face_mounting(model, params):
    data = _pose(model, {})
    for leg, sx, sy in LEGS:
        housing = model.geom(f'{leg}_j1_housing_motor')
        centre = data.geom_xpos[housing.id]
        radius, half_length = housing.size[:2]
        # Cylinder radius projects onto x; its axial half-length projects onto y.
        assert sx * centre[0] - radius - params['torso_lx'] / 2 == pytest.approx(
            params['assembly_clearance'], abs=2e-6)
        assert params['torso_ly'] / 2 - sy * centre[1] - half_length == pytest.approx(
            params['assembly_clearance'], abs=2e-6)
        assert centre[2] == pytest.approx(params['standing_height'], abs=2e-6)
        axis = data.xaxis[model.joint(f'{leg}_j1').id]
        np.testing.assert_allclose(axis, [0, sx, 0], atol=1e-12)
    left = data.geom_xpos[model.geom('fl_j1_housing_motor').id]
    right = data.geom_xpos[model.geom('fr_j1_housing_motor').id]
    assert left[1] - right[1] - 2 * model.geom('fl_j1_housing_motor').size[1] > 0


def test_j1_housings_clear_end_faces_and_fit_inside_torso_width(generated):
    model, params, _ = generated
    _assert_end_face_mounting(model, params)


def test_joint_motor_and_foot_inventory(generated):
    model, _, report = generated
    hinges = [model.joint(i).name for i in range(model.njnt)
              if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    assert set(hinges) == {f'{leg}_j{i}' for leg, _, _ in LEGS for i in (1, 2, 3)}
    assert not any('hip_roll' in name for name in hinges)
    assert sum(model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) == 4
    motors = [model.geom(i).name for i in range(model.ngeom)
              if model.geom(i).name.endswith('_motor')]
    assert len(motors) == 12
    assert np.all(model.dof_armature[6:] > 0)
    assert report['actuator_mass_audit']['n_joints'] == 12
    assert report['actuator_mass_audit']['emitted_over_charged'] == pytest.approx(1.0)
    assert report['actuator_packing']['n_actuator_geoms'] == 12
    assert report['actuator_packing']['n_exempt'] == 0
    assert report['actuator_packing']['n_overlapping_pairs'] == 0
    assert report['actuator_packing']['n_neutral_pose_pairs'] == 0
    assert report['actuator_in_structure']['inside_cm3'] == 0
    assert sum(model.body_mass) == pytest.approx(report['total_mass_kg'], abs=1e-4)
    assert report['xml_roundtrip']['relative_drift'] < 1e-5
    assert any('allow_hypothetical' in warning for warning in report['warnings'])


def test_motor_masses_are_attached_to_the_upstream_rigid_assembly(generated):
    model, _, _ = generated
    for leg, _, _ in LEGS:
        for owner, motor in [('j1_housing', 'j1_housing'),
                             ('j1_link', 'j1_link'), ('thigh_link', 'thigh_link')]:
            assert model.geom(f'{leg}_{motor}_motor').bodyid == model.body(f'{leg}_{owner}').id
        assert model.body(f'{leg}_j1_housing').jntnum[0] == 0
        assert model.body(f'{leg}_thigh_link').jntnum[0] == 0
        assert model.body(f'{leg}_shank_link').mass[0] > 0


@pytest.mark.parametrize('q', [(0, 0, 0), (0.3, 0.4, -0.8), (-0.25, -0.6, -1.1),
                               (0, 0, math.pi / 4), (0.3, 0.4, math.pi / 2)])
@pytest.mark.parametrize('leg,sx,sy', LEGS)
def test_forward_kinematics_matches_serial_chain(generated, leg, sx, sy, q):
    model, params, _ = generated
    data = _pose(model, {leg: q})
    expected = _analytical_positions(params, sx, sy, q)
    for index, point in enumerate(expected[:3], 1):
        jid = model.joint(f'{leg}_j{index}').id
        np.testing.assert_allclose(data.xanchor[jid], point, atol=2e-6)
    np.testing.assert_allclose(data.site_xpos[model.site(f'{leg}_foot').id],
                               expected[3], atol=2e-6)
    a1 = data.xaxis[model.joint(f'{leg}_j1').id]
    a2 = data.xaxis[model.joint(f'{leg}_j2').id]
    assert float(a1 @ a2) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize('knee', [-0.7, 0, math.pi / 4, math.pi / 2])
def test_equal_commands_produce_front_back_and_left_right_symmetry(generated, knee):
    model, _, _ = generated
    data = _pose(model, {leg: (0.2, 0.35, knee) for leg, _, _ in LEGS})
    fl = data.site_xpos[model.site('fl_foot').id]
    for leg, sx, sy in LEGS:
        np.testing.assert_allclose(data.site_xpos[model.site(f'{leg}_foot').id],
                                   [sx * fl[0], sy * fl[1], fl[2]], atol=1e-10)


def test_zero_pose_feet_are_tangent_to_floor(generated):
    model, params, _ = generated
    data = _pose(model, {})
    for leg, _, _ in LEGS:
        assert data.site_xpos[model.site(f'{leg}_foot').id, 2] == pytest.approx(
            params['foot_sphere_r'], abs=2e-6)


@pytest.mark.parametrize('leg,sx,sy', LEGS)
def test_j1_positive_collects_leg_and_j2_positive_abducts(generated, leg, sx, sy):
    model, _, _ = generated
    baseline = _pose(model, {})
    knee = model.joint(f'{leg}_j3').id
    pitch = _pose(model, {leg: (0.15, 0, 0)})
    yaw = _pose(model, {leg: (0, 0.15, 0)})
    assert sx * pitch.xanchor[knee, 0] < sx * baseline.xanchor[knee, 0]
    assert sy * yaw.xanchor[knee, 1] > sy * baseline.xanchor[knee, 1]


@pytest.mark.parametrize('q2', [-math.pi / 2, -0.4, 0, 0.7, math.pi / 2])
def test_j2_sweeps_knee_around_its_moving_axis(generated, q2):
    model, params, _ = generated
    data = _pose(model, {'fl': (0.3, q2, 0)})
    jid = model.joint('fl_j2').id
    offset = data.xanchor[model.joint('fl_j3').id] - data.xanchor[jid]
    axis = data.xaxis[jid]
    height = float(offset @ axis)
    radius = float(np.linalg.norm(offset - height * axis))
    assert radius == pytest.approx(params['upper_leg_link_length']
                                  * math.sin(params['thigh_cant_rad']), abs=2e-6)
    assert height == pytest.approx(-params['j2_output_offset']
                                  - params['upper_leg_link_length']
                                  * math.cos(params['thigh_cant_rad']), abs=2e-6)


@pytest.mark.parametrize('angle', [-0.8, 0, math.pi / 4, math.pi / 2])
def test_knee_rotation_keeps_joint_centre_and_thigh_fixed(generated, angle):
    model, params, _ = generated
    before = _pose(model, {'fl': (0.2, 0.4, 0)})
    after = _pose(model, {'fl': (0.2, 0.4, angle)})
    jid, sid = model.joint('fl_j3').id, model.site('fl_foot').id
    np.testing.assert_allclose(before.xanchor[jid], after.xanchor[jid], atol=1e-12)
    bid = model.body('fl_thigh_link').id
    np.testing.assert_allclose(before.xmat[bid], after.xmat[bid], atol=1e-12)
    # The foot travels the chord of a circle centred at the knee, on either side.
    assert np.linalg.norm(before.site_xpos[sid] - after.site_xpos[sid]) == pytest.approx(
        2 * params['lower_leg_link_length'] * abs(math.sin(angle / 2)), abs=1e-6)
    assert np.linalg.norm(after.site_xpos[sid] - after.xanchor[jid]) == pytest.approx(
        params['lower_leg_link_length'], abs=1e-6)


def test_viewer_knee_sliders_include_positive_quarter_turn(generated):
    pytest.importorskip('viser')
    pytest.importorskip('mjviser')
    from draft.editor.viewer import Viewer3D

    model, params, _ = generated
    assert params['j3_range'] == pytest.approx([-2.8, math.pi / 2])
    # Exercise the real slider scan/pose methods without starting a network server.
    viewer = Viewer3D.__new__(Viewer3D)
    viewer._mj_model = model
    viewer._mj_data = mujoco.MjData(model)
    viewer._scene_lock = threading.Lock()
    viewer._simulate = False
    viewer._scene = None
    viewer._scan_joints()
    knees = [j for j in viewer.joints() if j['name'].endswith('_j3')]
    assert len(knees) == 4
    for joint in knees:
        assert model.joint(joint['name']).limited[0]
        # Written MJCF numbers have six significant figures.
        assert [joint['lo'], joint['hi']] == pytest.approx([-2.8, math.pi / 2], abs=5e-6)
        viewer.set_joint(joint['qadr'], math.pi / 2)
    for joint in viewer.joints():
        if joint['name'].endswith('_j3'):
            assert joint['value'] == pytest.approx(math.pi / 2)


def test_structural_report_uses_actual_canted_joint_span(generated):
    model, params, report = generated
    data = _pose(model, {})
    measured = np.linalg.norm(data.xanchor[model.joint('fl_j3').id]
                              - data.xanchor[model.joint('fl_j2').id])
    assert measured == pytest.approx(params['thigh_joint_span'], abs=2e-6)
    for name in ('fl_j2_link_link', 'fl_thigh_link_link'):
        assert report['structural_links'][name]['joint_to_joint_m'] == pytest.approx(
            measured, abs=2e-6)


def test_short_abduction_link_is_rejected_by_packing_gate(tmp_path):
    with pytest.raises(FeasibilityViolation, match='packing'):
        RobotGenerator(robot_dir('galileo_x')).generate(
            tmp_path, params_override={'hip_abduction_link_length': 0.03})


def test_editor_selects_changes_and_regenerates_galileo_x(tmp_path):
    pytest.importorskip('viser')
    pytest.importorskip('mjviser')
    pytest.importorskip('websockets')
    from draft.editor.web.server import WebEditor
    from draft.generation.mjcf_assets import load_model

    source = robot_dir('galileo_x') / 'parameters.yaml'
    original = source.read_bytes()
    loaded = []

    def view(path):
        loaded.append(load_model(path))
        return True

    viewer = SimpleNamespace(port=0, simulate=False, joints=lambda: [],
                             load_model=view, set_status=lambda text: None)
    editor = WebEditor(viewer, repo_root())
    editor.generated_base = tmp_path

    async def exercise():
        await editor.handle({'type': 'select_robot', 'robot': 'galileo_x'})
        state = editor.state()
        assert state['robot'] == 'galileo_x'
        assert state['unsupported'] is None
        editable = {p['key'] for group in state['groups'] for p in group['params']}
        assert {'thigh_cant_deg', 'upper_leg_link_length', 'lower_leg_link_length'} <= editable
        await editor.handle({'type': 'generate'})
        assert editor.status['kind'] == 'ok'
        first_knee = _pose(loaded[-1], {}).xanchor[loaded[-1].joint('fl_j3').id].copy()
        await editor.handle({'type': 'patch', 'key': 'thigh_cant_deg', 'value': 45})
        await editor.handle({'type': 'patch', 'key': 'upper_leg_link_length', 'value': 0.28})
        await editor.handle({'type': 'patch', 'key': 'torso_lx', 'value': 0.36})
        await editor.handle({'type': 'patch', 'key': 'torso_ly', 'value': 0.28})
        await editor.handle({'type': 'generate'})
        assert editor.status['kind'] == 'ok'
        out = editor.session.folders.selected_dir
        params = yaml.safe_load((out / 'parameters_resolved.yaml').read_text(encoding='utf-8'))
        assert params['thigh_cant_rad'] == pytest.approx(math.pi / 4)
        assert params['upper_leg_link_length'] == 0.28
        assert params['torso_lx'] == 0.36
        assert params['torso_ly'] == 0.28
        model = loaded[-1]
        _assert_end_face_mounting(model, params)
        actual = _pose(model, {}).xanchor[model.joint('fl_j3').id]
        np.testing.assert_allclose(actual, _analytical_positions(params, 1, 1, (0, 0, 0))[2],
                                   atol=2e-6)
        assert np.linalg.norm(actual - first_knee) > 0.05

    asyncio.run(exercise())
    assert source.read_bytes() == original

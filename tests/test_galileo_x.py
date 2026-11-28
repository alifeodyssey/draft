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
    hip = _ry(-q[0])
    j2 = j1 + hip @ [0, sy * params['hip_abduction_link_length'], 0]
    yaw = hip @ _rz(-sy * q[1])
    mount = j2 + yaw @ [0, 0, -params['j2_output_offset']]
    thigh = yaw @ _ry(params['thigh_cant_rad'])
    knee = mount + thigh @ [0, 0, -params['upper_leg_link_length']]
    foot = knee + thigh @ _ry(q[2]) @ [0, sy * (params['shank_outboard_offset']
                                               + params['j4_outboard_offset']),
                                      -params['lower_leg_link_length']]
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
        np.testing.assert_allclose(axis, [0, -1, 0], atol=1e-12)
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
    assert set(hinges) == {f'{leg}_j{i}' for leg, _, _ in LEGS for i in (1, 2, 3, 4)}
    assert not any('hip_roll' in name for name in hinges)
    assert sum(model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) == 0
    assert model.nu == 4
    motors = [model.geom(i).name for i in range(model.ngeom)
              if model.geom(i).name.endswith('_motor')]
    assert len(motors) == 16
    assert np.all(model.dof_armature[6:] > 0)
    assert report['actuator_mass_audit']['n_joints'] == 16
    assert report['actuator_mass_audit']['emitted_over_charged'] == pytest.approx(1.0)
    assert report['actuator_packing']['n_actuator_geoms'] == 16
    assert report['actuator_packing']['n_exempt'] == 0
    assert report['actuator_packing']['n_overlapping_pairs'] == 0
    assert report['actuator_packing']['n_neutral_pose_pairs'] == 0
    assert report['actuator_in_structure']['inside_cm3'] == 0
    assert sum(model.body_mass) == pytest.approx(report['total_mass_kg'], abs=1e-4)
    assert report['xml_roundtrip']['relative_drift'] < 1e-5
    assert not any('allow_hypothetical' in warning for warning in report['warnings'])
    assert not report['actuator_checks']['errors']
    assert {name: cls['anchor'] for name, cls in report['actuator_checks']['classes'].items()} == {
        'J1': 'JM150', 'J2': 'JM100', 'J3': 'JM250', 'J4': 'AK70-10'}


def test_motor_masses_are_attached_to_the_upstream_rigid_assembly(generated):
    model, _, _ = generated
    for leg, _, _ in LEGS:
        for owner, motor in [('j1_housing', 'j1_housing'),
                             ('j1_link', 'j1_link'), ('thigh_link', 'thigh_link'),
                             ('j4_housing', 'j4_housing')]:
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
def test_equal_commands_keep_front_back_parallel_and_left_right_mirrored(generated, knee):
    model, params, _ = generated
    data = _pose(model, {leg: (0.2, 0.35, knee) for leg, _, _ in LEGS})
    fl = data.site_xpos[model.site('fl_foot').id]
    for leg, sx, sy in LEGS:
        np.testing.assert_allclose(data.site_xpos[model.site(f'{leg}_foot').id],
                                   [fl[0] + (sx - 1) * params['j1_mount_x'],
                                    sy * fl[1], fl[2]], atol=2e-6)


def test_zero_pose_feet_are_tangent_to_floor(generated):
    model, params, _ = generated
    data = _pose(model, {})
    for leg, _, _ in LEGS:
        assert data.site_xpos[model.site(f'{leg}_foot').id, 2] == pytest.approx(
            params['wheel_radius'], abs=2e-6)


@pytest.mark.parametrize('leg,sx,sy', LEGS)
def test_j1_positive_swings_forward_and_j2_positive_abducts(generated, leg, sx, sy):
    model, _, _ = generated
    baseline = _pose(model, {})
    knee = model.joint(f'{leg}_j3').id
    pitch = _pose(model, {leg: (0.15, 0, 0)})
    yaw = _pose(model, {leg: (0, 0.15, 0)})
    assert pitch.xanchor[knee, 0] > baseline.xanchor[knee, 0]
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
    assert height == pytest.approx(params['j2_output_offset']
                                  + params['upper_leg_link_length']
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
        math.hypot(params['lower_leg_link_length'],
                   params['shank_outboard_offset'] + params['j4_outboard_offset']), abs=1e-6)


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

    viewer = SimpleNamespace(port=0, simulate=False, has_standing_pose=False, joints=lambda: [],
                             load_model=view, set_status=lambda text: None)
    editor = WebEditor(viewer, repo_root())
    editor.generated_base = tmp_path

    async def exercise():
        await editor.handle({'type': 'select_robot', 'robot': 'galileo_x'})
        state = editor.state()
        assert state['robot'] == 'galileo_x'
        assert state['unsupported'] is None
        editable = {p['key'] for group in state['groups'] for p in group['params']}
        assert {'thigh_cant_deg', 'upper_leg_link_length', 'lower_leg_link_length',
                'standing_j1_deg', 'standing_j3_deg'} <= editable
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
        assert params['standing_j3_deg'] == -130
        model = loaded[-1]
        _assert_end_face_mounting(model, params)
        actual = _pose(model, {}).xanchor[model.joint('fl_j3').id]
        np.testing.assert_allclose(actual, _analytical_positions(params, 1, 1, (0, 0, 0))[2],
                                   atol=2e-6)
        assert np.linalg.norm(actual - first_knee) > 0.05

    asyncio.run(exercise())
    assert source.read_bytes() == original


def _standing(model):
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'standing')
    assert key >= 0
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return data


def test_standing_pose_has_rearward_knees_and_four_feet_on_floor(generated):
    model, params, _ = generated
    data = _standing(model)
    assert data.qpos[2] == pytest.approx(params['standing_pose_height'], abs=2e-6)
    assert model.qpos0[2] == pytest.approx(params['standing_height'], abs=2e-6)
    assert data.qpos[2] < model.qpos0[2]
    for leg, _, sy in LEGS:
        for index, expected in [(1, params['standing_j1_rad']), (2, 0),
                                (3, params['standing_j3_rad'])]:
            assert data.qpos[model.joint(f'{leg}_j{index}').qposadr[0]] == pytest.approx(
                expected, abs=5e-6)
        j2 = model.joint(f'{leg}_j2').id
        knee = data.xanchor[model.joint(f'{leg}_j3').id]
        foot = data.site_xpos[model.site(f'{leg}_foot').id]
        assert knee[0] < data.xanchor[j2, 0]
        assert foot[0] > knee[0]
        assert foot[2] == pytest.approx(params['wheel_radius'], abs=3e-6)
        # Reverse the spin sign to compare the upward motor-axis direction.
        axis = -sy * data.xaxis[j2]
        np.testing.assert_allclose(axis, [math.sin(-params['standing_j1_rad']), 0,
                                         math.cos(params['standing_j1_rad'])], atol=5e-6)


def test_standing_pose_preserves_actuator_and_torso_clearance(generated):
    from draft.generation import packing
    from draft.generation.spec_builder import SpecEmitter

    model, params, _ = generated
    data = _standing(model)
    motors = []
    for gi in range(model.ngeom):
        geom = model.geom(gi)
        if not geom.name.endswith('_motor'):
            continue
        body = int(model.geom_bodyid[gi])
        r, h = model.geom_size[gi, :2]
        motors.append({'name': geom.name, 'body': model.body(body).name,
                       'path': SpecEmitter._body_path(model, body),
                       'center': data.geom_xpos[gi], 'R': data.geom_xmat[gi],
                       'half': [r, r, h]})
    report = packing.check(motors)
    assert report['n_overlapping_pairs'] == 0
    assert report['n_neutral_pose_pairs'] == 0
    structure = packing.structure_overlap(model, data, params['torso_rho'])
    assert not packing.structure_violations(structure, ())


@pytest.mark.parametrize('overrides, expected_knee', [
    ({'torso_lx': 0.5, 'torso_ly': 0.32, 'torso_lz': 0.18,
      'upper_leg_link_length': 0.25, 'lower_leg_link_length': 0.24,
      'thigh_cant_deg': 20, 'standing_j1_deg': -15}, -70),
    ({'standing_j1_deg': -10, 'standing_j3_deg': -50}, -50),
])
def test_standing_parameters_regenerate_keyframe_and_ground_height(tmp_path, overrides, expected_knee):
    RobotGenerator(robot_dir('galileo_x')).generate(tmp_path, params_override=overrides)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / 'galileo_x.xml'))
    data = _standing(model)
    params = yaml.safe_load((tmp_path / 'parameters_resolved.yaml').read_text(encoding='utf-8'))
    assert params['standing_j3_deg'] == expected_knee
    for leg, _, _ in LEGS:
        assert math.degrees(data.qpos[model.joint(f'{leg}_j3').qposadr[0]]) == pytest.approx(
            expected_knee, abs=5e-4)
        assert data.site_xpos[model.site(f'{leg}_foot').id, 2] == pytest.approx(
            params['wheel_radius'], abs=3e-6)


@pytest.mark.parametrize('overrides, joint', [
    ({'standing_j1_deg': 200}, 'fl_j1'),
    ({'standing_j3_deg': -170}, 'fl_j3'),
])
def test_out_of_range_standing_pose_is_rejected(tmp_path, overrides, joint):
    with pytest.raises(ValueError, match=f"standing.*{joint}.*outside range"):
        RobotGenerator(robot_dir('galileo_x')).generate(tmp_path, params_override=overrides)
    assert not (tmp_path / 'galileo_x.xml').exists()


def test_viewer_load_reset_zero_reload_and_stop_restore_named_stance(generated, tmp_path, monkeypatch):
    pytest.importorskip('viser')
    pytest.importorskip('mjviser')
    from draft.editor import viewer as viewer_module
    from draft.editor.web.server import WebEditor

    model, _, _ = generated
    scene = SimpleNamespace(update_from_mjdata=lambda data: None)
    monkeypatch.setattr(viewer_module, 'ViserMujocoScene', lambda *a, **kw: scene)
    monkeypatch.setattr(viewer_module.mjcf_assets, 'load_model', lambda path: model)
    viewer = viewer_module.Viewer3D.__new__(viewer_module.Viewer3D)
    viewer.server = SimpleNamespace(scene=SimpleNamespace(reset=lambda: None))
    viewer.port = 0
    viewer._scene_lock = threading.Lock()
    viewer._simulate = False
    assert viewer.load_model(tmp_path / 'galileo_x.xml')
    assert viewer.has_standing_pose
    expected = _standing(model).qpos.copy()
    np.testing.assert_allclose(viewer._mj_data.qpos, expected)
    editor = WebEditor(viewer, repo_root())
    editor.generated_base = tmp_path

    async def exercise():
        await editor.handle({'type': 'select_robot', 'robot': 'galileo_x'})
        assert editor.state()['has_standing_pose']
        await editor.handle({'type': 'reset_pose', 'zero': True})
        np.testing.assert_allclose(viewer._mj_data.qpos, model.qpos0)
        await editor.handle({'type': 'reset_pose'})
        np.testing.assert_allclose(viewer._mj_data.qpos, expected)
        viewer.set_joint(model.joint('fl_j1').qposadr[0], 0.5)
        await editor.handle({'type': 'simulate', 'on': True})
        await editor.handle({'type': 'simulate', 'on': False})
        np.testing.assert_allclose(viewer._mj_data.qpos, expected)

    asyncio.run(exercise())
    viewer.reset_pose(zero=True)
    assert viewer.load_model(tmp_path / 'galileo_x.xml')
    np.testing.assert_allclose(viewer._mj_data.qpos, expected)


def test_viewer_without_standing_keyframe_keeps_mechanical_zero():
    pytest.importorskip('viser')
    pytest.importorskip('mjviser')
    from draft.editor.viewer import Viewer3D

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    data.qpos[2] = 5
    Viewer3D._reset_data(model, data)
    np.testing.assert_array_equal(data.qpos, model.qpos0)


@pytest.mark.parametrize('angles', [(0, 0, 0), (-0.35, 0, -1.75), (0.4, 0.6, -0.9)])
def test_wheels_follow_local_knee_pitch_and_clear_the_outboard_shank(generated, angles):
    model, params, _ = generated
    data = _pose(model, {leg: (*angles, 9 * math.pi) for leg, _, _ in LEGS})
    for leg, _, sy in LEGS:
        knee, wheel = (model.joint(f'{leg}_j{i}').id for i in (3, 4))
        axis = data.xaxis[knee]
        np.testing.assert_allclose(data.xaxis[wheel], axis, atol=1e-12)
        if angles[1] == 0:
            np.testing.assert_allclose(axis, [0, 1, 0], atol=1e-12)
        shank = data.xpos[model.body(f'{leg}_shank_link').id]
        assert sy * float((shank - data.xanchor[knee]) @ axis) == pytest.approx(
            params['shank_outboard_offset'], abs=2e-6)
        assert sy * float((data.xanchor[wheel] - shank) @ axis) == pytest.approx(
            params['j4_outboard_offset'], abs=2e-6)
        distance = mujoco.mj_geomDistance(model, data, model.geom(f'{leg}_wheel_geom').id,
                                         model.geom(f'{leg}_shank_link_link').id, 0.1, None)
        assert distance == pytest.approx(params['assembly_clearance'], abs=2e-6)


def test_spin_changes_only_wheel_rotor_and_all_drives_are_independent(generated):
    model, params, report = generated
    before = _pose(model, {'fl': (0.2, 0.3, -0.9, 0)})
    after = _pose(model, {'fl': (0.2, 0.3, -0.9, 8 * math.pi + 0.4)})
    for body in ('fl_j3_output', 'fl_shank_link', 'fl_j4_housing'):
        bid = model.body(body).id
        np.testing.assert_allclose(before.xpos[bid], after.xpos[bid], atol=1e-12)
        np.testing.assert_allclose(before.xmat[bid], after.xmat[bid], atol=1e-12)
    np.testing.assert_allclose(before.xanchor, after.xanchor, atol=1e-12)
    assert not np.allclose(before.xmat[model.body('fl_wheel').id],
                           after.xmat[model.body('fl_wheel').id])
    assert report['mass_kg']['wheel'] == pytest.approx(4 * params['wheel_mass'])
    expected = np.zeros(model.nv)
    for index, (leg, _, _) in enumerate(LEGS):
        jid = model.joint(f'{leg}_j4').id
        aid = model.actuator(f'{leg}_j4_drive').id
        assert not model.jnt_limited[jid]
        assert model.actuator_trnid[aid, 0] == jid
        after.ctrl[aid] = index + 1
        expected[model.jnt_dofadr[jid]] = index + 1
        assert model.body(f'{leg}_wheel').mass[0] == pytest.approx(params['wheel_mass'])
        for suffix in ('stripe', 'marker'):
            geom = model.geom(f'{leg}_wheel_{suffix}_geom')
            assert model.geom_contype[geom.id] == model.geom_conaffinity[geom.id] == 0
            assert model.body(int(geom.bodyid[0])).mass[0] == 0
    mujoco.mj_forward(model, after)
    np.testing.assert_allclose(after.qfrc_actuator, expected)
    for suffix in ('j3_output', 'shank_link', 'j4_housing'):
        assert report['structural_links'][f'fl_{suffix}_link']['joint_to_joint_m'] == pytest.approx(
            params['lower_leg_link_length'], abs=2e-6)


def test_web_wheel_preview_has_unbounded_phase_and_reset_clears_speeds(generated):
    pytest.importorskip('viser')
    pytest.importorskip('mjviser')
    from draft.editor.viewer import Viewer3D
    from draft.editor.web.server import WebEditor

    model, _, _ = generated
    viewer = Viewer3D.__new__(Viewer3D)
    viewer.port = 0
    viewer._mj_model, viewer._mj_data = model, _standing(model)
    viewer._scene, viewer._scene_lock, viewer._simulate = None, threading.Lock(), False
    viewer._scan_joints()
    editor = WebEditor(viewer, repo_root())
    before = viewer._mj_data.qpos.copy()

    async def exercise():
        await editor.handle({'type': 'set_preview_velocity', 'joint': 'fl_j4', 'velocity': 5})
        await editor.handle({'type': 'set_preview_velocity', 'joint': 'rr_j4', 'velocity': -4})
        viewer._step_preview(4)
        assert viewer._mj_data.qpos[model.joint('fl_j4').qposadr[0]] == 20
        assert viewer._mj_data.qpos[model.joint('rr_j4').qposadr[0]] == -16
        for leg, _, _ in LEGS:
            for i in (1, 2, 3):
                adr = model.joint(f'{leg}_j{i}').qposadr[0]
                assert viewer._mj_data.qpos[adr] == before[adr]
        assert all(j['lo'] is None and j['hi'] is None for j in viewer.joints() if j['continuous'])
        await editor.handle({'type': 'stop_preview'})
        frozen = viewer._mj_data.qpos.copy()
        viewer._step_preview(1)
        np.testing.assert_array_equal(viewer._mj_data.qpos, frozen)
        await editor.handle({'type': 'reset_pose'})
        np.testing.assert_array_equal(viewer._mj_data.qpos, before)
        await editor.handle({'type': 'set_preview_velocity', 'joint': 'fl_j4', 'velocity': 2})
        await editor.handle({'type': 'simulate', 'on': True})
        assert not any(j['preview_velocity'] for j in viewer.joints())
        await editor.handle({'type': 'simulate', 'on': False})
        assert not any(j['preview_velocity'] for j in viewer.joints())
        await editor.handle({'type': 'reset_pose', 'zero': True})
        np.testing.assert_array_equal(viewer._mj_data.qpos, model.qpos0)

    asyncio.run(exercise())
    for name, speed, message in [('fl_j3', 1, 'unlimited'), ('fl_j4', math.nan, 'finite'),
                                  ('fl_j4', 51, 'velocity limit')]:
        with pytest.raises(ValueError, match=message):
            viewer.set_preview_velocity(name, speed)


def test_sitting_path_checks_real_geometries_through_body_floor_contact(generated):
    from draft.generation.wheel_leg_inspection import inspect_sitting
    model, params, _ = generated
    result = inspect_sitting(model, params)
    report = result.report
    assert report['valid']
    assert report['samples'] > 300
    assert report['max_joint_step_deg'] <= 0.5 + 1e-9
    assert report['max_base_step_m'] <= 0.001 + 1e-9
    assert report['minimum']['parts']['distance_m'] >= params['assembly_clearance'] - 2e-6
    assert report['minimum']['hardware_floor']['distance_m'] == pytest.approx(0.020, abs=1e-6)
    assert report['end_torso_floor_gap_m'] == pytest.approx(0, abs=2e-6)
    assert list(report['end_wheel_floor_gaps_m'].values()) == pytest.approx([0.0990241] * 4, abs=2e-6)
    assert report['end_joint_degrees']['fl_j1'] == pytest.approx(-140)
    assert report['end_joint_degrees']['fl_j3'] == pytest.approx(-145)


@pytest.mark.parametrize('overrides,message', [
    ({'wheel_radius': 0.01}, 'wheel_radius'),
    ({'wheel_width': 0.01}, 'wheel_width'),
    ({'wheel_mass': 0}, 'wheel_mass'),
    ({'sitting_j1_deg': -200}, 'outside range'),
    ({'sitting_j3_deg': -170}, 'outside range'),
    ({'torso_lz': 0.05}, 'hardware_floor:'),
])
def test_invalid_wheel_or_sitting_design_reports_the_failure(generated, overrides, message):
    from draft.generation.wheel_leg_inspection import inspect_sitting
    model, params, _ = generated
    with pytest.raises(ValueError, match=message):
        inspect_sitting(model, {**params, **overrides})

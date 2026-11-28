"""Catalogue frontier precision and editor/generator agreement for named motors."""
import copy
import math

import mujoco
import pytest
import yaml

from draft.editor.core.session import EditorSession, MotorInputs
from draft.editor.shared import discover_robot_types
from draft.generation.generator import size_actuators, RobotGenerator
from draft.paths import repo_root
from draft.robots import robot_dir
from draft.trends.feasibility import FeasibilityViolation, motor_fits
from draft.trends.motor_catalog import resolve_catalog_motors
from draft.trends.motor_solve import design_class, frontier_of


PARTS = {
    'J1': ('JM150', 150.0, 150.6*math.pi/30, 30.0, 0.80, 0.0375, 0.080),
    'J2': ('JM100', 100.0, 159.3*math.pi/30, 36.0, 0.59, 0.0375, 0.055),
    'J3': ('JM250', 250.0, 145.0*math.pi/30, 25.0, 1.15, 0.0450, 0.080),
    'J4': ('AK70-10', 24.0, 32.46, 10.0, 0.521, 0.0445, 0.05025),
}


def _params():
    return RobotGenerator(robot_dir('galileo_x')).params


def _rs04_params():
    return {'A_motor': 'RS04', 'A_motor_mode': 'torque_speed', 'A_motor_effort': 120.,
            'A_motor_velocity': 20.94, 'A_motor_gear': 9., 'allow_hypothetical': False}


def _session(tmp_path):
    return EditorSession(discover_robot_types(repo_root())['galileo_x'], tmp_path)


def test_rs04_does_not_exceed_its_own_rounded_catalogue_frontier():
    laws = motor_fits()
    frozen = copy.deepcopy(laws.frontier)
    design = design_class('A', _rs04_params())
    assert not design.errors
    assert design.densities()['torque_density']['status'] == 'above_p90'
    assert frontier_of('QDD')['torque_density_Nm_per_kg']['max'] == pytest.approx(120 / 1.42, rel=1e-14)
    assert design.ceilings()['by_torque_density'] == pytest.approx(120, rel=1e-12)
    assert laws.frontier == frozen  # paper summaries and fit coefficients stay untouched


def test_a_genuine_small_increase_over_rs04_frontier_is_still_rejected():
    params = {**_rs04_params(), 'A_motor_effort': 120.0001}
    with pytest.raises(FeasibilityViolation, match='torque density'):
        size_actuators(resolve_catalog_motors(params))


def test_named_motor_inputs_and_editor_sizing_preserve_the_four_requested_points():
    inputs = MotorInputs(PARTS, _params())
    assert tuple(inputs.classes) == ('J1', 'J2', 'J3', 'J4')
    for cls, (part, effort, velocity, gear, mass, radius, length) in PARTS.items():
        design = inputs.design(cls)
        assert inputs.collect()[f'{cls}_motor'] == part
        assert design.sized['anchor'] == part
        assert (design.point.tau, design.point.omega, design.point.gear) == pytest.approx((effort, velocity, gear))
        assert design.mass == pytest.approx(mass)
        assert (design.sized['r'], design.sized['L']) == pytest.approx((radius, length))
        assert not design.errors


def test_editor_resolves_catalogue_only_inputs():
    inputs = MotorInputs(('A',), {'A_motor': 'RS04'})
    assert inputs.design('A').sized['anchor'] == 'RS04'
    assert inputs.design('A').mass == pytest.approx(1.42)


def test_editor_model_name_changes_and_numeric_loads_do_not_keep_stale_anchors(tmp_path):
    session = _session(tmp_path)
    session.private['_J2_motor_catalog'] = 'RS03'
    session.set('J2_motor', 'AK70-10')
    assert session.motors.design('J2').sized['anchor'] == 'AK70-10'
    assert session.motors.design('J3').sized['anchor'] == 'JM250'
    direct = design_class('J2', resolve_catalog_motors(session.collect()))
    assert direct.sized == session.motors.design('J2').sized
    session.set('J2_motor', '')
    collected = session.collect()
    assert collected['J2_motor'] == ''
    assert '_J2_motor_catalog' not in collected
    assert design_class('J2', resolve_catalog_motors(collected)).sized['anchor'] is None
    session.load_params({'J1_motor_mode': 'torque_speed', 'J1_motor_effort': 80,
                         'J1_motor_velocity': 20, 'J1_motor_gear': 9})
    assert session.get('J1_motor') == ''
    assert session.motors.design('J1').sized['anchor'] is None
    assert session.collect()['J1_motor'] == ''


@pytest.mark.parametrize('name', ['', 'RS03'])
def test_public_name_overrides_stale_saved_catalogue_metadata(name):
    params = resolve_catalog_motors({'A_motor': name, 'A_motor_effort': 60,
                                    '_A_motor_catalog': 'RS04',
                                    'A_motor_catalog_name': 'RS04'})
    assert params.get('_A_motor_catalog') == ('RS03' if name else None)
    assert params.get('A_motor_catalog_name') == ('RS03' if name else None)


def test_editor_save_load_and_regenerate_matches_the_generator(tmp_path):
    session = _session(tmp_path)
    out = session.save_as('catalogue-roundtrip')
    assert (out / 'galileo_x.xml').is_file()
    saved = yaml.safe_load((out / 'parameters_from_gui.yaml').read_text(encoding='utf-8'))
    restored = _session(tmp_path)
    restored.folders.select(out.name)
    assert restored.load_selected_folder() is not None
    assert restored.motors.collect() == session.motors.collect()
    result = restored.generate()
    assert result.ok, result.error
    params = yaml.safe_load((out / 'parameters_resolved.yaml').read_text(encoding='utf-8'))
    report = yaml.safe_load((out / 'feasibility_report.yaml').read_text(encoding='utf-8'))
    for cls, (part, effort, velocity, gear, mass, radius, length) in PARTS.items():
        assert saved[f'{cls}_motor'] == part
        display = restored.motors.design(cls)
        generated = report['actuator_checks']['classes'][cls]
        assert generated['anchor'] == part
        assert generated['mass_kg'] == pytest.approx(display.mass, abs=1e-6)
        assert (params[f'{cls}_motor_r'], params[f'{cls}_motor_L']) == pytest.approx((radius, length))
    assert params['wheel_radius'] == pytest.approx(0.0645)
    assert params['wheel_width'] == pytest.approx(0.06025)
    assert report['actuator_mass_audit']['charged_by_motor_law_kg'] == pytest.approx(12.244)
    assert report['actuator_mass_audit']['emitted_over_charged'] == 1
    assert not report['actuator_checks']['errors']
    assert not params['allow_hypothetical']
    model = mujoco.MjModel.from_xml_path(str(out / 'galileo_x.xml'))
    data = mujoco.MjData(model)
    for leg in ('fl', 'fr', 'rl', 'rr'):
        jid, aid = model.joint(f'{leg}_j4').id, model.actuator(f'{leg}_j4_drive').id
        assert not model.jnt_limited[jid]
        assert list(model.actuator_ctrlrange[aid]) == [-24, 24]
        assert model.numeric(f'{leg}_j4_velocity_limit').data[0] == 32.46
        data.ctrl[aid] = 100
    mujoco.mj_forward(model, data)
    assert list(data.actuator_force) == [24, 24, 24, 24]

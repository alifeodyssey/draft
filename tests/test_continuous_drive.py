"""Optional output-shaft drives and unlimited hinges without robot special cases."""
import math

import mujoco
import numpy as np
import pytest

from draft.generation.spec_builder import SpecEmitter


PARAMS = {f'L_motor_{key}': value for key, value in {
    'r': 0.02, 'L': 0.03, 'rho': 1000, 'effort': 10, 'velocity': 20,
    'friction': 0.01, 'damping': 0.001, 'armature': 0.001,
}.items()}


def _emitter(joint):
    emitter = SpecEmitter('drive_bench')
    emitter.emit({'name': 'rotor', 'type': 'geom', 'joint': {
        'name': 'spin', 'axis': 'y', 'motor_class': 'L', **joint},
        'geometry': {'shape': 'cylinder', 'axis': 'y', 'r': 0.05, 'l': 0.03, 'mass': 0.2}}, PARAMS)
    return emitter


def test_unlimited_drive_roundtrip_spins_both_ways_and_limits_output_torque():
    emitter = _emitter({'limited': False, 'actuator': 'motor'})
    emitter.compile()
    model = mujoco.MjModel.from_xml_string(emitter.spec.to_xml())
    assert not model.jnt_limited[0]
    assert model.nu == 1
    assert model.actuator(0).name == 'spin_drive'
    np.testing.assert_array_equal(model.actuator_gear[0], [1, 0, 0, 0, 0, 0])
    for direction in (-1, 1):
        data = mujoco.MjData(model)
        data.ctrl[0] = direction * 0.2
        for _ in range(1000):
            mujoco.mj_step(model, data)
        assert direction * data.qpos[0] > 4 * math.pi
    data = mujoco.MjData(model)
    data.ctrl[0] = 100
    mujoco.mj_forward(model, data)
    assert data.actuator_force[0] == pytest.approx(10)
    assert data.qfrc_actuator[0] == pytest.approx(10)


def test_existing_limited_joint_does_not_acquire_a_drive():
    model = _emitter({'range': [-1, 1]}).compile()
    assert model.nu == 0
    assert model.jnt_limited[0]


@pytest.mark.parametrize('joint, message', [
    ({'limited': False, 'range': [-1, 1]}, 'omit range'),
    ({'limited': True}, 'needs range'),
    ({'limited': 'false'}, 'true or false'),
    ({'limited': False, 'actuator': 'velocity'}, 'unsupported actuator'),
])
def test_invalid_continuous_declarations_are_explicit(joint, message):
    with pytest.raises(ValueError, match=message):
        _emitter(joint)

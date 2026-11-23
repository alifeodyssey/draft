"""Named keyframes resolve against MuJoCo's compiled joint addresses."""

import copy

import mujoco
import numpy as np
import pytest

from draft.generation.spec_builder import SpecEmitter


TREE = {
    'name': 'base', 'type': 'root', 'pos': [0, 0, 1], 'children': [
        {'name': 'ballast', 'type': 'geom',
         'geometry': {'shape': 'sphere', 'r': 0.1, 'mass': 1}},
        {'name': 'link', 'type': 'geom', 'pos': [0, 0, 0.2],
         'joint': {'name': 'hinge', 'axis': 'y', 'range': [-1, 1], 'motor_class': 'L'},
         'geometry': {'shape': 'sphere', 'r': 0.03, 'mass': 0.1}},
    ],
}
PARAMS = {f'L_motor_{key}': value for key, value in {
    'r': 0.02, 'L': 0.03, 'rho': 1000, 'effort': 10, 'velocity': 20,
    'friction': 0.01, 'damping': 0.001, 'armature': 0.001,
}.items()}


def _emitter(frames):
    tree = copy.deepcopy(TREE)
    tree['keyframes'] = frames
    emitter = SpecEmitter('keyframe_test')
    emitter.emit(tree, PARAMS)
    return emitter


def test_named_keyframes_roundtrip_without_changing_zero_geometry_or_mass():
    baseline = _emitter([]).compile()
    emitter = _emitter([
        {'name': 'standing', 'joints': {'hinge': 0.5,
                                        'base_freejoint': [0.1, 0.2, 0.5, 1, 0, 0, 0]}},
        {'name': 'rest', 'joints': {'hinge': -0.25}},
    ])
    model = emitter.compile()
    assert emitter.compile().nkey == 2
    written = mujoco.MjModel.from_xml_string(emitter.to_string())
    for attr in ('qpos0', 'body_pos', 'body_quat', 'body_mass', 'body_inertia', 'geom_size'):
        np.testing.assert_allclose(getattr(model, attr), getattr(baseline, attr), atol=1e-12)
    data = mujoco.MjData(written)
    mujoco.mj_resetDataKeyframe(written, data, written.key('standing').id)
    np.testing.assert_allclose(data.qpos[:7], [0.1, 0.2, 0.5, 1, 0, 0, 0])
    assert data.qpos[written.joint('hinge').qposadr[0]] == 0.5
    mujoco.mj_resetDataKeyframe(written, data, written.key('rest').id)
    np.testing.assert_allclose(data.qpos[:7], baseline.qpos0[:7])


@pytest.mark.parametrize('frames, message', [
    ([{'name': 'bad', 'joints': {'missing': 0}}], 'unknown joint'),
    ([{'name': 'bad', 'joints': {'hinge': [0, 1]}}], 'expected 1 finite'),
    ([{'name': 'bad', 'joints': {'hinge': float('nan')}}], 'finite'),
    ([{'name': 'bad', 'joints': {'base_freejoint': [0, 0, 1]}}], 'expected 7 finite'),
    ([{'name': 'bad', 'joints': {'base_freejoint': [0, 0, 1, 2, 0, 0, 0]}}], 'unit length'),
    ([{'name': 'bad', 'joints': {'hinge': 1.01}}], 'outside range'),
    ([{'name': 'same'}, {'name': 'same'}], 'duplicate keyframe'),
    ([{'name': ''}], 'nonempty name'),
    ([{'name': 'bad', 'joints': [0]}], 'mapping'),
    ({'standing': {}}, 'must be a list'),
])
def test_invalid_keyframes_are_rejected_with_context(frames, message):
    with pytest.raises(ValueError, match=message):
        _emitter(frames).compile()

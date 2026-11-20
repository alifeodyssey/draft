"""Structural sizing uses rigid transforms, including fixed canted mounts."""

import math

import pytest

from draft.generation.expr_eval import resolve
from draft.generation.generator import _assign_structural_mass
from draft.trends.feasibility import link_fits


def test_trigonometric_expressions_use_radians():
    angle = resolve('${degrees * pi / 180}', {'degrees': 30.0})
    assert resolve('${sin(angle)}', {'angle': angle}) == pytest.approx(0.5)
    assert resolve('${cos(angle)}', {'angle': angle}) == pytest.approx(math.sqrt(3) / 2)


@pytest.mark.parametrize('tilt', [(0.0, math.pi / 6, 0.0), (0.25, 0.5, 0.7)])
@pytest.mark.parametrize('base_euler', [(0.0, 0.0, 0.0), (0.3, -0.4, 0.6)])
def test_canted_span_includes_rotated_offsets(tilt, base_euler):
    length, axial_offset = 0.22, 0.04
    member = {
        'name': 'canted_thigh', 'pos': [0, 0, -axial_offset], 'euler': list(tilt),
        'geometry': {'link_l': length, 'link_mass_class': 'thigh'},
        'children': [{'name': 'knee', 'pos': [0, 0, -length],
                      'joint': {'name': 'j3', 'axis': 'y'}}],
    }
    tree = {
        'name': 'root', 'pos': [0.1, -0.2, 0.5], 'euler': list(base_euler),
        'children': [{'name': 'yaw', 'joint': {'name': 'j2', 'axis': 'z'},
                      'children': [member]}],
    }
    _assign_structural_mass(tree, link_fits('quadruped'), 'quadruped')
    # For intrinsic xyz, the rotated -z member has vertical projection
    # -L*cos(rx)*cos(ry). A common base rotation cannot change this distance.
    expected = math.sqrt(length ** 2 + axial_offset ** 2
                         + 2 * length * axial_offset * math.cos(tilt[0])
                         * math.cos(tilt[1]))
    basis = member['geometry']['link_mass_basis']
    assert basis['length_m'] == pytest.approx(expected, abs=1e-12)
    assert basis['length_m'] != pytest.approx(0.26, abs=1e-5)
    law = link_fits('quadruped').structural_mass('thigh', expected)
    assert member['geometry']['link_mass_kg'] == pytest.approx(law['mass_kg'])


@pytest.mark.parametrize('distal_axis', ['y', '-y'])
def test_parallel_axes_are_compared_in_a_common_frame(distal_axis):
    member = {
        'name': 'proximal', 'euler': [0.2, -0.3, 0.4],
        'joint': {'name': 'proximal', 'axis': '-x'},
        'geometry': {'link_l': math.sqrt(0.05), 'link_mass_class': 'thigh'},
        'children': [{
            'name': 'distal', 'pos': [0.2, 0.1, 0],
            'euler': [0, 0, math.pi / 2],
            'joint': {'name': 'distal', 'axis': distal_axis},
        }],
    }
    tree = {'name': 'root', 'children': [member]}
    _assign_structural_mass(tree, link_fits('quadruped'), 'quadruped')
    # The distal local y axis becomes +/- proximal x after its 90-degree turn.
    # The 0.2 m slide along those parallel axes must not increase their spacing.
    assert member['geometry']['link_mass_basis']['length_m'] == pytest.approx(0.1)
